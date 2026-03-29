-- ==========================================================================
-- Genetic Algorithm Brute Force - Super Mario Bros 1-1
-- ==========================================================================
-- Standalone FCEUX Lua script. No Python needed.
-- Evolves button-press sequences to beat World 1-1.
--
-- Usage: Load ROM in FCEUX, create save state in slot 1 at 1-1 start,
--        then load this script via File > Lua > New Lua Script Window.
--
-- HOW IT WORKS:
--   1. A "genome" is a sequence of action IDs (one per decision frame).
--   2. Each genome is played out in the emulator using savestate resets.
--   3. Fitness = max X distance reached.
--   4. Best genomes are selected, crossed over, and mutated to make the next generation.
--   5. Repeat until a genome completes the level.
-- ==========================================================================

-- ========================
-- CONFIGURATION
-- ========================
local CONFIG = {
    POPULATION_SIZE = 50,       -- Genomes per generation
    GENOME_LENGTH   = 900,      -- Max actions per genome (~3600 frames at skip=4)
    FRAME_SKIP      = 4,        -- Repeat each action for N frames
    MUTATION_RATE   = 0.02,     -- Base chance of mutating each gene
    MUTATION_RATE_MAX = 0.15,   -- Max mutation rate when stuck
    CROSSOVER_RATE  = 0.7,      -- Chance of crossover vs clone
    ELITE_COUNT     = 5,        -- Top N genomes kept unchanged
    TOURNAMENT_SIZE = 5,        -- Tournament selection pool size
    TIMEOUT_FRAMES  = 300,      -- Frames without progress before giving up
    STAGNATION_GENS = 5,        -- Gens without improvement before boosting mutation
    INJECTION_RATE  = 0.1,      -- Fraction of pop replaced with random genomes when stuck
    SAVESTATE_SLOT  = 1,        -- Save state slot for level start
    TARGET_X        = 3160,     -- X position to consider "completed"
}

-- ========================
-- ACTION MAPPING (same as DQN)
-- ========================
local ACTIONS = {
    [0]  = {},                                          -- No action
    [1]  = {right = true},                              -- Right
    [2]  = {left = true},                               -- Left
    [3]  = {A = true},                                  -- Jump
    [4]  = {right = true, A = true},                    -- Right + Jump
    [5]  = {left = true, A = true},                     -- Left + Jump
    [6]  = {B = true},                                  -- Run
    [7]  = {right = true, B = true},                    -- Right + Run
    [8]  = {left = true, B = true},                     -- Left + Run
    [9]  = {right = true, A = true, B = true},          -- Right + Jump + Run
    [10] = {left = true, A = true, B = true},           -- Left + Jump + Run
    [11] = {down = true},                               -- Down
}
local NUM_ACTIONS = 12  -- 0-11

-- ========================
-- MEMORY ADDRESSES
-- ========================
local ADDR = {
    MARIO_X_PAGE = 0x006D,
    MARIO_X_POS  = 0x0086,
    MARIO_Y      = 0x00CE,
    MARIO_VEL_X  = 0x0057,
    MARIO_VEL_Y  = 0x009F,
    LIVES        = 0x075A,
    TIME_H       = 0x07F8,
    FLOAT_STATE  = 0x001D,
    PLAYER_STATE = 0x000E,
}

-- ========================
-- HELPER FUNCTIONS
-- ========================

local function get_mario_x()
    return memory.readbyte(ADDR.MARIO_X_PAGE) * 256 + memory.readbyte(ADDR.MARIO_X_POS)
end

local function is_dead()
    -- Player state 0x0B = dying, 0x06 = dead
    local state = memory.readbyte(ADDR.PLAYER_STATE)
    return state == 0x0B or state == 0x06
end

local function get_timer()
    return memory.readbyte(ADDR.TIME_H) * 100
         + memory.readbyte(0x07F9) * 10
         + memory.readbyte(0x07FA)
end

-- ========================
-- GENOME OPERATIONS
-- ========================

local function random_genome()
    local genome = {}
    -- Bias toward forward movement: 60% right-ish actions, 40% random
    for i = 1, CONFIG.GENOME_LENGTH do
        if math.random() < 0.6 then
            -- Forward-biased: pick from right, right+jump, right+run, right+jump+run
            local forward = {1, 4, 7, 9, 9, 9}  -- Extra weight on right+jump+run
            genome[i] = forward[math.random(#forward)]
        else
            genome[i] = math.random(0, NUM_ACTIONS - 1)
        end
    end
    return genome
end

local function mutate(genome, rate)
    rate = rate or CONFIG.MUTATION_RATE
    local new = {}
    for i = 1, #genome do
        if math.random() < rate then
            -- Mutate this gene
            if math.random() < 0.7 then
                -- Random action
                new[i] = math.random(0, NUM_ACTIONS - 1)
            else
                -- Shift to adjacent action (small mutation)
                new[i] = math.max(0, math.min(NUM_ACTIONS - 1, genome[i] + math.random(-1, 1)))
            end
        else
            new[i] = genome[i]
        end
    end
    return new
end

local function crossover(parent1, parent2)
    if math.random() > CONFIG.CROSSOVER_RATE then
        -- No crossover, clone parent1
        local child = {}
        for i = 1, #parent1 do child[i] = parent1[i] end
        return child
    end
    
    -- Single-point crossover
    local point = math.random(1, math.min(#parent1, #parent2))
    local child = {}
    for i = 1, point do
        child[i] = parent1[i]
    end
    for i = point + 1, math.max(#parent1, #parent2) do
        child[i] = (parent2[i] or parent1[i])
    end
    return child
end

local function tournament_select(population, fitnesses)
    local best_idx = nil
    local best_fit = -1
    for t = 1, CONFIG.TOURNAMENT_SIZE do
        local idx = math.random(1, #population)
        if fitnesses[idx] > best_fit then
            best_fit = fitnesses[idx]
            best_idx = idx
        end
    end
    return population[best_idx]
end

-- ========================
-- EVALUATE A GENOME
-- ========================

-- Checkpoint X positions for segment-based splicing (evenly spaced landmarks)
local CHECKPOINTS = { 400, 800, 1200, 1600, 2000, 2400, 2800 }

local function evaluate_genome(genome, show_overlay)
    -- Run at maximum speed when not showing overlay, normal when showing
    if show_overlay then
        emu.speedmode("normal")
    else
        emu.speedmode("turbo")  -- Skip rendering, run as fast as CPU allows
    end
    
    -- Load save state to reset to level start
    savestate.load(savestate.object(CONFIG.SAVESTATE_SLOT))
    
    -- Settling delay: let emulator sync after save state load
    for i = 1, 30 do emu.frameadvance() end
    
    local max_x = 0
    local frames_stuck = 0
    local last_x = 0
    local total_frames = 0
    local completed = false
    
    -- Track gene index when each checkpoint is first reached
    local checkpoint_genes = {}  -- checkpoint_x -> gene_idx
    local next_cp = 1            -- index into CHECKPOINTS to check next
    
    for gene_idx = 1, #genome do
        local action = ACTIONS[genome[gene_idx]] or {}
        
        -- Execute action for FRAME_SKIP frames
        for f = 1, CONFIG.FRAME_SKIP do
            joypad.set(1, action)
            
            -- Draw overlay on last frame of each action if requested
            if show_overlay and f == CONFIG.FRAME_SKIP then
                local x = get_mario_x()
                local progress = math.min(100, x / 3168 * 100)
                gui.drawbox(0, 0, 100, 22, "#000000C0", "#000000C0")
                gui.text(2, 2,  string.format("X:%d (%.0f%%)", x, progress), "#44FF44")
                gui.text(2, 12, string.format("Gene:%d/%d", gene_idx, #genome), "#FFFFFF")
            end
            
            emu.frameadvance()
            total_frames = total_frames + 1
        end
        
        -- Check position
        local current_x = get_mario_x()
        if current_x > max_x then
            max_x = current_x
            frames_stuck = 0
        else
            frames_stuck = frames_stuck + CONFIG.FRAME_SKIP
        end
        last_x = current_x
        
        -- Record checkpoint arrival (gene index when first crossing each checkpoint)
        while next_cp <= #CHECKPOINTS and current_x >= CHECKPOINTS[next_cp] do
            checkpoint_genes[CHECKPOINTS[next_cp]] = gene_idx
            next_cp = next_cp + 1
        end
        
        -- Check termination conditions
        if is_dead() then
            break
        end
        
        if current_x >= CONFIG.TARGET_X then
            completed = true
            -- Mark remaining checkpoints as reached at this gene
            while next_cp <= #CHECKPOINTS do
                checkpoint_genes[CHECKPOINTS[next_cp]] = gene_idx
                next_cp = next_cp + 1
            end
            break
        end
        
        if frames_stuck > CONFIG.TIMEOUT_FRAMES then
            break
        end
        
        if get_timer() <= 0 then
            break
        end
    end
    
    -- Fitness: distance + time bonus for completion
    local fitness = max_x
    if completed then
        fitness = fitness + 10000 + (400 - total_frames / 60) * 10  -- Bonus for speed
    end
    
    return fitness, max_x, completed, total_frames, checkpoint_genes
end

-- Build a spliced genome from the fastest segment between each checkpoint pair.
-- segment_db: list of { genome=..., checkpoints={cp_x -> gene_idx} } for completers
-- Returns a new genome or nil if not enough data.
local function splice_fastest_segments(segment_db)
    if #segment_db < 2 then return nil end  -- Need at least 2 genomes to splice
    
    -- Build ordered checkpoint list: 0 (start) + CHECKPOINTS + end
    local cps = {0}
    for _, cp in ipairs(CHECKPOINTS) do cps[#cps + 1] = cp end
    
    -- For each segment [cp_k, cp_{k+1}], find the genome that traversed it in fewest genes
    local best_segments = {}  -- { genome_idx, start_gene, end_gene } per segment
    
    for seg = 1, #cps do
        local cp_start = cps[seg]
        local cp_end = cps[seg + 1]  -- nil for last segment (to end of genome)
        
        local best_cost = 999999
        local best_idx = nil
        
        for di, data in ipairs(segment_db) do
            local g_start = (cp_start == 0) and 1 or (data.checkpoints[cp_start] or nil)
            local g_end = cp_end and (data.checkpoints[cp_end] or nil) or #data.genome
            
            if g_start and g_end then
                local cost = g_end - g_start
                if cost < best_cost then
                    best_cost = cost
                    best_idx = di
                end
            end
        end
        
        if best_idx then
            local data = segment_db[best_idx]
            local g_start = (cp_start == 0) and 1 or data.checkpoints[cp_start]
            local g_end = cp_end and data.checkpoints[cp_end] or #data.genome
            best_segments[#best_segments + 1] = {
                idx = best_idx, start_gene = g_start, end_gene = g_end,
                cp_start = cp_start, cp_end = cp_end or "END"
            }
        end
    end
    
    if #best_segments == 0 then return nil end
    
    -- Assemble the spliced genome
    local spliced = {}
    for _, seg in ipairs(best_segments) do
        local src = segment_db[seg.idx].genome
        for g = seg.start_gene, seg.end_gene do
            if src[g] then
                spliced[#spliced + 1] = src[g]
            end
        end
    end
    
    if #spliced < 10 then return nil end  -- sanity check
    
    print(string.format("  [SPLICE] Built Frankenstein genome: %d genes from %d segments",
        #spliced, #best_segments))
    for _, seg in ipairs(best_segments) do
        print(string.format("    Seg %s->%s: genome #%d genes %d-%d (%d genes)",
            tostring(seg.cp_start), tostring(seg.cp_end),
            seg.idx, seg.start_gene, seg.end_gene, seg.end_gene - seg.start_gene))
    end
    
    return spliced
end

-- ========================
-- MAIN GA LOOP
-- ========================

-- Try to load an existing genome from file (for seeding)
local function load_existing_genome(filename)
    local f = io.open(filename, "r")
    if not f then return nil end
    local content = f:read("*all")
    f:close()
    local loader = loadstring(content)
    if not loader then return nil end
    local ok, genome = pcall(loader)
    if ok and genome and #genome > 0 then
        print(string.format("  Loaded existing genome from %s (%d actions)", filename, #genome))
        return genome
    end
    return nil
end

-- Save genome to file (overwrites if faster)
local function save_genome(filename, genome, gen, fitness, max_x, frames, label)
    local f = io.open(filename, "w")
    if not f then return end
    f:write(string.format("-- %s genome for SMB 1-1\n", label or "Winning"))
    f:write(string.format("-- Generation: %d, Fitness: %d, Max X: %d", gen, fitness, max_x))
    if frames then f:write(string.format(", Frames: %d (%.1fs)", frames, frames / 60)) end
    f:write("\n")
    f:write("return {")
    for i = 1, #genome do
        if i > 1 then f:write(",") end
        if i % 50 == 1 then f:write("\n  ") end
        f:write(tostring(genome[i]))
    end
    f:write("\n}\n")
    f:close()
    print(string.format("  Saved %s to %s", label or "genome", filename))
end

local function run_ga()
    math.randomseed(os.time())
    
    -- Initialize population
    local population = {}
    local fitnesses = {}
    local generation = 0
    local all_time_best_fitness = 0
    local all_time_best_genome = nil
    local all_time_best_x = 0
    local level_completed = false
    local gens_without_improvement = 0
    local current_mutation_rate = CONFIG.MUTATION_RATE
    
    print("=== Genetic Algorithm - SMB 1-1 ===")
    print(string.format("Population: %d, Genome length: %d, Frame skip: %d",
        CONFIG.POPULATION_SIZE, CONFIG.GENOME_LENGTH, CONFIG.FRAME_SKIP))
    
    -- Try to seed from existing genomes
    local seed_genome = load_existing_genome("winning_genome.txt")
    local seed_fast = load_existing_genome("winning_genome_fast.txt")
    
    -- If we have an existing winner, evaluate it first
    if seed_genome then
        print("Evaluating existing winning genome...")
        local fit, mx, completed, frames = evaluate_genome(seed_genome, false)
        if completed then
            print(string.format("  Existing genome completes level! (x=%d, frames=%d)", mx, frames))
            all_time_best_genome = seed_genome
            all_time_best_fitness = fit
            all_time_best_x = mx
            level_completed = true
            -- Skip Phase 1 entirely, go straight to speed optimization
            return seed_genome, 0
        else
            print(string.format("  Existing genome reaches x=%d but doesn't complete", mx))
        end
    end
    
    print("Initializing first generation...")
    
    -- Seed population: existing genomes + random
    local seed_count = 0
    if seed_genome then
        -- Add existing genome and mutated variants
        population[1] = seed_genome
        seed_count = 1
        for i = 2, math.min(10, CONFIG.POPULATION_SIZE) do
            local child = {}
            for j = 1, #seed_genome do child[j] = seed_genome[j] end
            child = mutate(child, 0.05)  -- Light mutation of seed
            population[i] = child
            seed_count = seed_count + 1
        end
    end
    if seed_fast then
        seed_count = seed_count + 1
        population[seed_count] = seed_fast
    end
    
    -- Fill remaining with random genomes
    for i = seed_count + 1, CONFIG.POPULATION_SIZE do
        population[i] = random_genome()
    end
    
    while not level_completed do
        generation = generation + 1
        fitnesses = {}
        
        local gen_best_fitness = 0
        local gen_best_x = 0
        local gen_best_idx = 1
        
        -- Evaluate each genome
        for i = 1, #population do
            -- Show overlay for best genome replay and first genome of each gen
            local show = (i == 1)
            
            local fitness, max_x, completed, frames = evaluate_genome(population[i], show)
            fitnesses[i] = fitness
            
            if fitness > gen_best_fitness then
                gen_best_fitness = fitness
                gen_best_x = max_x
                gen_best_idx = i
            end
            
            if completed then
                level_completed = true
                all_time_best_genome = population[i]
                all_time_best_fitness = fitness
                all_time_best_x = max_x
                print(string.format("!!! LEVEL COMPLETED !!! Gen %d, Genome %d, Frames: %d", 
                    generation, i, frames))
                break
            end
            
            -- Progress update every 10 genomes
            if i % 10 == 0 then
                gui.drawbox(0, 224, 255, 240, "#000000", "#000000")
                gui.text(2, 226, string.format("Gen %d: Evaluating %d/%d  Best: %d",
                    generation, i, #population, gen_best_x), "#FFFF44")
                emu.frameadvance()
            end
        end
        
        -- Update all-time best
        if gen_best_fitness > all_time_best_fitness then
            all_time_best_fitness = gen_best_fitness
            all_time_best_x = gen_best_x
            gens_without_improvement = 0
            current_mutation_rate = CONFIG.MUTATION_RATE  -- Reset mutation rate
            -- Deep copy best genome
            all_time_best_genome = {}
            for i = 1, #population[gen_best_idx] do
                all_time_best_genome[i] = population[gen_best_idx][i]
            end
        else
            gens_without_improvement = gens_without_improvement + 1
        end
        
        -- Anti-stagnation: ramp up mutation when stuck at a pit
        if gens_without_improvement >= CONFIG.STAGNATION_GENS then
            current_mutation_rate = math.min(CONFIG.MUTATION_RATE_MAX,
                CONFIG.MUTATION_RATE + (gens_without_improvement - CONFIG.STAGNATION_GENS) * 0.01)
        end
        
        -- Print generation summary
        local avg_fitness = 0
        for i = 1, #fitnesses do avg_fitness = avg_fitness + fitnesses[i] end
        avg_fitness = avg_fitness / #fitnesses
        
        local stag_msg = ""
        if gens_without_improvement >= CONFIG.STAGNATION_GENS then
            stag_msg = string.format(" [STUCK x%d, mut=%.0f%%]",
                gens_without_improvement, current_mutation_rate * 100)
        end
        
        print(string.format("Gen %d: best_x=%d avg=%.0f all_time_best=%d (%.1f%%)%s",
            generation, gen_best_x, avg_fitness, all_time_best_x,
            all_time_best_x / 3168 * 100, stag_msg))
        
        if level_completed then break end
        
        -- === SELECTION AND BREEDING ===
        
        -- Sort by fitness (descending) for elitism
        local sorted_indices = {}
        for i = 1, #population do sorted_indices[i] = i end
        table.sort(sorted_indices, function(a, b) return fitnesses[a] > fitnesses[b] end)
        
        local new_population = {}
        
        -- Elitism: keep top N unchanged
        for i = 1, CONFIG.ELITE_COUNT do
            local elite = population[sorted_indices[i]]
            local copy = {}
            for j = 1, #elite do copy[j] = elite[j] end
            new_population[i] = copy
        end
        
        -- Inject random genomes when stagnating (fresh blood to escape local optima)
        local inject_count = 0
        if gens_without_improvement >= CONFIG.STAGNATION_GENS then
            inject_count = math.floor(CONFIG.POPULATION_SIZE * CONFIG.INJECTION_RATE)
        end
        
        -- Fill rest with tournament selection + crossover + mutation (dynamic rate)
        for i = CONFIG.ELITE_COUNT + 1, CONFIG.POPULATION_SIZE do
            if inject_count > 0 then
                -- Inject a completely random genome to break out of local optima
                new_population[i] = random_genome()
                inject_count = inject_count - 1
            else
                local parent1 = tournament_select(population, fitnesses)
                local parent2 = tournament_select(population, fitnesses)
                local child = crossover(parent1, parent2)
                child = mutate(child, current_mutation_rate)
                new_population[i] = child
            end
        end
        
        population = new_population
        
        -- Replay best genome with overlay
        if generation % 5 == 0 and all_time_best_genome then
            print(string.format("  Replaying all-time best (x=%d)...", all_time_best_x))
            evaluate_genome(all_time_best_genome, true)
        end
    end
    
    -- === VICTORY LAP WITH ANALYSIS ===
    if all_time_best_genome then
        print("=== VICTORY! Analyzing winning genome ===")
        emu.speedmode("normal")
        
        -- Analysis replay: log every action at every position
        savestate.load(savestate.object(CONFIG.SAVESTATE_SLOT))
        for i = 1, 30 do emu.frameadvance() end  -- settle delay
        
        local csv = io.open("genome_analysis.csv", "w")
        csv:write("gene_idx,frame,action_id,action_name,mario_x,mario_y,vel_x,vel_y,")
        csv:write("float_state,on_ground,delta_x\n")
        
        local prev_x = get_mario_x()
        local total_frames = 0
        local jumps = {}
        local jump_start_x = nil
        local jump_start_gene = nil
        local jump_peak_y = 999
        local prev_y = memory.readbyte(ADDR.MARIO_Y)
        
        local ACTION_NAMES = {
            [0]="NOOP",[1]="Right",[2]="Left",[3]="Jump",
            [4]="Right+Jump",[5]="Left+Jump",[6]="Run",
            [7]="Right+Run",[8]="Left+Run",[9]="Right+Jump+Run",
            [10]="Left+Jump+Run",[11]="Down"
        }
        
        for gene_idx = 1, #all_time_best_genome do
            local action_id = all_time_best_genome[gene_idx]
            local action = ACTIONS[action_id] or {}
            local aname = ACTION_NAMES[action_id] or "?"
            
            for f = 1, CONFIG.FRAME_SKIP do
                joypad.set(1, action)
                local x = get_mario_x()
                local progress = math.min(100, x / 3168 * 100)
                gui.drawbox(0, 0, 145, 22, "#000000C0", "#000000C0")
                gui.text(2, 2,  string.format("X:%d (%.1f%%) Gene:%d", x, progress, gene_idx), "#44FF44")
                gui.text(2, 12, string.format("%s  Frame:%d", aname, total_frames), "#FFFFFF")
                emu.frameadvance()
                total_frames = total_frames + 1
            end
            
            local x = get_mario_x()
            local y = memory.readbyte(ADDR.MARIO_Y)
            local vx = memory.readbyte(ADDR.MARIO_VEL_X)
            if vx > 127 then vx = vx - 256 end
            local vy = memory.readbyte(ADDR.MARIO_VEL_Y)
            if vy > 127 then vy = vy - 256 end
            local float_st = memory.readbyte(ADDR.FLOAT_STATE)
            local on_gnd = (float_st == 0) and 1 or 0
            local dx = x - prev_x
            
            csv:write(string.format("%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d\n",
                gene_idx, total_frames, action_id, aname, x, y, vx, vy, float_st, on_gnd, dx))
            
            -- Track jumps
            if float_st ~= 0 then
                if not jump_start_x then
                    jump_start_x = prev_x
                    jump_start_gene = gene_idx
                    jump_peak_y = y
                end
                if y < jump_peak_y then jump_peak_y = y end
            else
                if jump_start_x then
                    table.insert(jumps, {
                        start_x = jump_start_x, end_x = x,
                        dist = x - jump_start_x,
                        height = prev_y - jump_peak_y,
                        genes = jump_start_gene .. "-" .. gene_idx
                    })
                    jump_start_x = nil
                end
            end
            
            prev_x = x
            prev_y = y
            
            if is_dead() or x >= 3100 then break end
        end
        csv:close()
        print("Saved: genome_analysis.csv")
        
        -- Write summary
        local sum = io.open("genome_analysis_summary.txt", "w")
        sum:write("=== WINNING GENOME ANALYSIS ===\n")
        sum:write(string.format("Total genes used: %d, Frames: %d (%.1fs)\n\n",
            #all_time_best_genome, total_frames, total_frames/60))
        
        sum:write("=== JUMP LOG ===\n")
        sum:write(string.format("%-6s %-8s %-8s %-8s %-8s %-10s\n",
            "#", "StartX", "EndX", "Dist", "Height", "Genes"))
        sum:write(string.rep("-", 55) .. "\n")
        for i, j in ipairs(jumps) do
            sum:write(string.format("%-6d %-8d %-8d %-8d %-8d %s\n",
                i, j.start_x, j.end_x, j.dist, j.height, j.genes))
        end
        
        sum:write("\n=== PIT-CROSSING JUMPS (dist>30, height>5) ===\n")
        for i, j in ipairs(jumps) do
            if j.dist > 30 and j.height > 5 then
                sum:write(string.format("  Jump %d: x=%d->%d (dist=%d, h=%d) genes %s\n",
                    i, j.start_x, j.end_x, j.dist, j.height, j.genes))
            end
        end
        
        sum:close()
        print("Saved: genome_analysis_summary.txt")
        
        -- Normal replay x2 for viewing pleasure
        print("=== Replaying 2 more times ===")
        for replay = 1, 2 do
            evaluate_genome(all_time_best_genome, true)
        end
        
        -- Save winning genome to file
        save_genome("winning_genome.txt", all_time_best_genome,
            generation, all_time_best_fitness, all_time_best_x, total_frames, "Winning")
    end
    
    print("=== Phase 1 Complete (Level Completion) ===")
    return all_time_best_genome, generation
end

-- ========================
-- PHASE 2: SPEED OPTIMIZATION
-- ========================
-- Takes a winning genome and evolves it for minimum frames.
-- Only genomes that still complete the level are considered.
-- Fitness = 50000 - total_frames (higher = faster completion).

local function optimize_speed(seed_genome, seed_generation)
    print("\n" .. string.rep("=", 60))
    print("PHASE 2: SPEED OPTIMIZATION")
    print(string.rep("=", 60))
    print("Evolving for minimum completion time...")
    print(string.format("Seed genome: %d actions, from generation %d\n", #seed_genome, seed_generation))
    
    local SPEED_CONFIG = {
        POPULATION_SIZE = 200,
        MAX_GENERATIONS = 100,       -- Max optimization generations
        -- Tiered mutation rates (applied per-genome, not uniform)
        MUTATION_LIGHT  = 0.015,     -- 60% of pop: gentle tweaks that preserve completion
        MUTATION_MEDIUM = 0.04,      -- 25% of pop: moderate exploration
        MUTATION_HEAVY  = 0.10,      -- 15% of pop: aggressive exploration (most will die)
        ELITE_COUNT     = 10,        -- Keep more elites with larger pop
        STAGNATION_GENS = 8,         -- Gens without improvement before boosting mutation
    }
    
    -- Helper: pick a tiered mutation rate for position in breeding order
    local function tiered_rate(slot_frac, stagnation_boost)
        local base
        if slot_frac < 0.60 then
            base = SPEED_CONFIG.MUTATION_LIGHT
        elseif slot_frac < 0.85 then
            base = SPEED_CONFIG.MUTATION_MEDIUM
        else
            base = SPEED_CONFIG.MUTATION_HEAVY
        end
        return base + stagnation_boost
    end
    
    -- Initialize population: seed genome + mutated variants
    local population = {}
    local slot = 0  -- tracks how many slots filled
    
    -- Try to load existing fast genome as an additional seed
    local existing_fast = load_existing_genome("winning_genome_fast.txt")
    if existing_fast then
        print("Evaluating existing fast genome...")
        local ef_fit, ef_x, ef_done, ef_frames = evaluate_genome(existing_fast, false)
        if ef_done then
            print(string.format("  Existing fast genome: %d frames (%.1fs)", ef_frames, ef_frames / 60))
            slot = slot + 1
            population[slot] = existing_fast
            -- Also add mutated variants of the fast genome
            for i = 1, math.min(5, SPEED_CONFIG.POPULATION_SIZE - slot) do
                local copy = {}
                for j = 1, #existing_fast do copy[j] = existing_fast[j] end
                copy = mutate(copy, 0.02)
                slot = slot + 1
                population[slot] = copy
            end
        else
            print("  Existing fast genome no longer completes -- ignoring")
        end
    end
    
    -- Clone seed as elite (if not already added)
    for i = 1, SPEED_CONFIG.ELITE_COUNT do
        slot = slot + 1
        local copy = {}
        for j = 1, #seed_genome do copy[j] = seed_genome[j] end
        population[slot] = copy
    end
    
    -- Fill rest with tiered mutation
    for i = slot + 1, SPEED_CONFIG.POPULATION_SIZE do
        local child = {}
        for j = 1, #seed_genome do child[j] = seed_genome[j] end
        local frac = (i - slot) / (SPEED_CONFIG.POPULATION_SIZE - slot)
        local rate = tiered_rate(frac, 0)
        child = mutate(child, rate)
        population[i] = child
    end
    
    local best_frames = 999999
    local best_genome = nil
    local best_x = 0
    local best_gen = 0
    local gens_without_improvement = 0
    
    -- If we loaded an existing fast genome, use its frames as the baseline
    if existing_fast then
        local _, ef_x, ef_done, ef_frames = evaluate_genome(existing_fast, false)
        if ef_done and ef_frames < best_frames then
            best_frames = ef_frames
            best_x = ef_x
            best_gen = 0
            best_genome = {}
            for j = 1, #existing_fast do best_genome[j] = existing_fast[j] end
            print(string.format("  Baseline from existing fast genome: %d frames (%.1fs)", best_frames, best_frames / 60))
        end
    end
    
    -- Segment database: keeps checkpoint timing from all completers (rolling window)
    local segment_db = {}
    local SEGMENT_DB_MAX = 50  -- Keep best 50 completers for splicing
    
    for gen = 1, SPEED_CONFIG.MAX_GENERATIONS do
        local fitnesses = {}
        local gen_best_frames = 999999
        local gen_best_idx = 1
        local completions = 0
        
        for i = 1, #population do
            local fitness, max_x, completed, frames, cp_genes = evaluate_genome(population[i], false)
            
            if completed then
                completions = completions + 1
                -- Speed fitness: lower frames = higher fitness
                fitnesses[i] = 50000 - frames
                
                if frames < gen_best_frames then
                    gen_best_frames = frames
                    gen_best_idx = i
                end
                
                if frames < best_frames then
                    best_frames = frames
                    best_x = max_x
                    best_gen = gen
                    best_genome = {}
                    for j = 1, #population[i] do best_genome[j] = population[i][j] end
                    gens_without_improvement = 0
                end
                
                -- Record checkpoint data for segment splicing
                -- Diversity filter: don't store if frame count is within 20 of an existing entry
                local dominated = false
                for _, entry in ipairs(segment_db) do
                    if math.abs(entry.frames - frames) < 20 then
                        dominated = true
                        break
                    end
                end
                if not dominated then
                    local genome_copy = {}
                    for j = 1, #population[i] do genome_copy[j] = population[i][j] end
                    segment_db[#segment_db + 1] = {
                        genome = genome_copy,
                        checkpoints = cp_genes,
                        frames = frames,
                        gen = gen
                    }
                    -- Keep only the fastest completers in the DB
                    if #segment_db > SEGMENT_DB_MAX then
                        table.sort(segment_db, function(a, b) return a.frames < b.frames end)
                        segment_db[#segment_db] = nil  -- remove slowest
                    end
                end
            else
                -- Non-completing genomes get low fitness based on distance
                fitnesses[i] = max_x * 0.1  -- much lower than any completion
            end
        end
        
        -- Stagnation tracking
        if gen_best_frames >= best_frames then
            gens_without_improvement = gens_without_improvement + 1
        end
        local stagnation_boost = 0
        if gens_without_improvement >= SPEED_CONFIG.STAGNATION_GENS then
            stagnation_boost = math.min(0.05, (gens_without_improvement - SPEED_CONFIG.STAGNATION_GENS) * 0.005)
        end
        
        local stag_msg = ""
        if gens_without_improvement >= SPEED_CONFIG.STAGNATION_GENS then
            stag_msg = string.format(" [STUCK x%d, boost=+%.1f%%]",
                gens_without_improvement, stagnation_boost * 100)
        end
        
        print(string.format("Speed Gen %d: completions=%d/%d, best=%d (%.1fs), all_time=%d (%.1fs), segs=%d%s",
            gen, completions, #population,
            gen_best_frames == 999999 and 0 or gen_best_frames,
            gen_best_frames == 999999 and 0 or gen_best_frames / 60,
            best_frames == 999999 and 0 or best_frames,
            best_frames == 999999 and 0 or best_frames / 60,
            #segment_db, stag_msg))
        
        -- If no completions in this generation, we've drifted too far
        if completions == 0 then
            print("  WARNING: No completions this gen, reverting to best known")
            -- Re-seed from best known with very light mutation
            for i = SPEED_CONFIG.ELITE_COUNT + 1, #population do
                local child = {}
                for j = 1, #best_genome do child[j] = best_genome[j] end
                child = mutate(child, 0.015)  -- very light to recover completions
                population[i] = child
            end
        else
            -- Breed next generation
            local sorted_indices = {}
            for i = 1, #population do sorted_indices[i] = i end
            table.sort(sorted_indices, function(a, b) return fitnesses[a] > fitnesses[b] end)
            
            local new_pop = {}
            -- Keep elites
            for i = 1, SPEED_CONFIG.ELITE_COUNT do
                local e = population[sorted_indices[i]]
                local c = {}
                for j = 1, #e do c[j] = e[j] end
                new_pop[i] = c
            end
            
            -- Every 5 gens, try segment splicing
            local splice_slot = SPEED_CONFIG.ELITE_COUNT + 1
            if gen % 5 == 0 and #segment_db >= 3 then
                local spliced = splice_fastest_segments(segment_db)
                if spliced then
                    -- Inject spliced genome + tiered mutated variants
                    new_pop[splice_slot] = spliced
                    splice_slot = splice_slot + 1
                    -- Light variant
                    local v1 = {}; for j = 1, #spliced do v1[j] = spliced[j] end
                    new_pop[splice_slot] = mutate(v1, 0.01); splice_slot = splice_slot + 1
                    -- Medium variant
                    local v2 = {}; for j = 1, #spliced do v2[j] = spliced[j] end
                    new_pop[splice_slot] = mutate(v2, 0.03); splice_slot = splice_slot + 1
                    -- Heavy variant
                    local v3 = {}; for j = 1, #spliced do v3[j] = spliced[j] end
                    new_pop[splice_slot] = mutate(v3, 0.08); splice_slot = splice_slot + 1
                    -- Trimmed variant (cut 5-15% off tail)
                    local v4 = {}; for j = 1, #spliced do v4[j] = spliced[j] end
                    local trim_len = math.max(100, math.floor(#v4 * (0.85 + math.random() * 0.10)))
                    local v4t = {}; for j = 1, trim_len do v4t[j] = v4[j] end
                    new_pop[splice_slot] = v4t; splice_slot = splice_slot + 1
                end
            end
            
            -- Breed rest with tiered mutation
            local breed_count = SPEED_CONFIG.POPULATION_SIZE - splice_slot + 1
            for i = splice_slot, SPEED_CONFIG.POPULATION_SIZE do
                local p1 = tournament_select(population, fitnesses)
                local p2 = tournament_select(population, fitnesses)
                local child = crossover(p1, p2)
                
                -- Tiered mutation: position in breeding order determines aggressiveness
                local frac = (i - splice_slot) / math.max(1, breed_count)
                local rate = tiered_rate(frac, stagnation_boost)
                child = mutate(child, rate)
                
                -- Smart trimming: try cutting 5-20% off tail for faster completion
                -- More frequent when stagnating
                local trim_chance = 0.10 + (stagnation_boost > 0 and 0.15 or 0)
                if math.random() < trim_chance then
                    local trim_pct = 0.80 + math.random() * 0.15  -- keep 80-95%
                    local trim_len = math.max(100, math.floor(#child * trim_pct))
                    local trimmed = {}
                    for j = 1, trim_len do trimmed[j] = child[j] end
                    new_pop[i] = trimmed
                else
                    new_pop[i] = child
                end
            end
            population = new_pop
        end
    end
    
    -- === SPEED VICTORY ===
    if best_genome then
        print(string.format("\n=== SPEED OPTIMIZED: %d frames (%.1fs) ===",
            best_frames, best_frames / 60))
        
        -- Replay at normal speed
        emu.speedmode("normal")
        evaluate_genome(best_genome, true)
        
        -- Save as fast genome
        local fitness = 50000 - best_frames
        save_genome("winning_genome_fast.txt", best_genome,
            best_gen, fitness, best_x, best_frames, "Speed-optimized")
        
        -- Also replace winning_genome.txt -- faster completion is the better genome
        save_genome("winning_genome.txt", best_genome,
            best_gen, fitness, best_x, best_frames, "Winning (speed-optimized)")
    else
        print("Speed optimization failed -- no completions found")
    end
end

-- ========================
-- START
-- ========================
print("=== SMB 1-1 Genetic Algorithm ===")
print("Phase 1: Find a completion")
print("Phase 2: Optimize for speed")
print("Make sure save state slot 1 has World 1-1 start!")
print("Starting in 2 seconds...")

-- Small delay to let user see the message
for i = 1, 120 do emu.frameadvance() end

local winner, gen = run_ga()
if winner then
    optimize_speed(winner, gen)
end

print("\n=== ALL PHASES COMPLETE ===")
