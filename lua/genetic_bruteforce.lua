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
    MUTATION_RATE   = 0.02,     -- Chance of mutating each gene
    CROSSOVER_RATE  = 0.7,      -- Chance of crossover vs clone
    ELITE_COUNT     = 5,        -- Top N genomes kept unchanged
    TOURNAMENT_SIZE = 5,        -- Tournament selection pool size
    TIMEOUT_FRAMES  = 300,      -- Frames without progress before giving up
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

local function mutate(genome)
    local new = {}
    for i = 1, #genome do
        if math.random() < CONFIG.MUTATION_RATE then
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

local function evaluate_genome(genome, show_overlay)
    -- Run at maximum speed when not showing overlay, normal when showing
    if show_overlay then
        emu.speedmode("normal")
    else
        emu.speedmode("turbo")  -- Skip rendering, run as fast as CPU allows
    end
    
    -- Load save state to reset to level start
    savestate.load(savestate.object(CONFIG.SAVESTATE_SLOT))
    
    local max_x = 0
    local frames_stuck = 0
    local last_x = 0
    local total_frames = 0
    local completed = false
    
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
        
        -- Check termination conditions
        if is_dead() then
            break
        end
        
        if current_x >= CONFIG.TARGET_X then
            completed = true
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
    
    return fitness, max_x, completed, total_frames
end

-- ========================
-- MAIN GA LOOP
-- ========================

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
    
    print("=== Genetic Algorithm - SMB 1-1 ===")
    print(string.format("Population: %d, Genome length: %d, Frame skip: %d",
        CONFIG.POPULATION_SIZE, CONFIG.GENOME_LENGTH, CONFIG.FRAME_SKIP))
    print("Initializing first generation...")
    
    for i = 1, CONFIG.POPULATION_SIZE do
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
            -- Deep copy best genome
            all_time_best_genome = {}
            for i = 1, #population[gen_best_idx] do
                all_time_best_genome[i] = population[gen_best_idx][i]
            end
        end
        
        -- Print generation summary
        local avg_fitness = 0
        for i = 1, #fitnesses do avg_fitness = avg_fitness + fitnesses[i] end
        avg_fitness = avg_fitness / #fitnesses
        
        print(string.format("Gen %d: best_x=%d avg=%.0f all_time_best=%d (%.1f%%)",
            generation, gen_best_x, avg_fitness, all_time_best_x,
            all_time_best_x / 3168 * 100))
        
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
        
        -- Fill rest with tournament selection + crossover + mutation
        for i = CONFIG.ELITE_COUNT + 1, CONFIG.POPULATION_SIZE do
            local parent1 = tournament_select(population, fitnesses)
            local parent2 = tournament_select(population, fitnesses)
            local child = crossover(parent1, parent2)
            child = mutate(child)
            new_population[i] = child
        end
        
        population = new_population
        
        -- Replay best genome with overlay
        if generation % 5 == 0 and all_time_best_genome then
            print(string.format("  Replaying all-time best (x=%d)...", all_time_best_x))
            evaluate_genome(all_time_best_genome, true)
        end
    end
    
    -- === VICTORY LAP ===
    if all_time_best_genome then
        print("=== VICTORY! Replaying winning genome at normal speed ===")
        emu.speedmode("normal")
        for replay = 1, 3 do
            evaluate_genome(all_time_best_genome, true)
        end
        
        -- Save winning genome to file
        local f = io.open("winning_genome.txt", "w")
        if f then
            f:write("-- Winning genome for SMB 1-1\n")
            f:write(string.format("-- Generation: %d, Fitness: %d, Max X: %d\n", 
                generation, all_time_best_fitness, all_time_best_x))
            f:write("return {")
            for i = 1, #all_time_best_genome do
                if i > 1 then f:write(",") end
                if i % 50 == 1 then f:write("\n  ") end
                f:write(tostring(all_time_best_genome[i]))
            end
            f:write("\n}\n")
            f:close()
            print("Winning genome saved to winning_genome.txt")
        end
    end
    
    print("=== GA Complete ===")
end

-- ========================
-- START
-- ========================
print("=== SMB 1-1 Genetic Algorithm ===")
print("Make sure save state slot 1 has World 1-1 start!")
print("Starting in 2 seconds...")

-- Small delay to let user see the message
for i = 1, 120 do emu.frameadvance() end

run_ga()
