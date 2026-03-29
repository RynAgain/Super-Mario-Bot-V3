-- ==========================================================================
-- NEAT (NeuroEvolution of Augmenting Topologies) - Super Mario Bros 1-1
-- ==========================================================================
-- Standalone FCEUX Lua script. No Python needed.
-- Evolves neural network topologies to play World 1-1.
--
-- Usage: Load ROM in FCEUX, create save state in slot 1 at 1-1 start,
--        then load this script via File > Lua > New Lua Script Window.
--
-- HOW IT WORKS:
--   1. Each genome encodes a neural network as a list of connection genes.
--   2. 13 input neurons read game state; 6 output neurons choose buttons.
--   3. Networks grow in complexity through structural mutations (add node,
--      add connection) tracked by innovation numbers for crossover alignment.
--   4. Speciation protects novel topologies from being overwhelmed.
--   5. Fitness = max X distance. Completion yields a large bonus.
-- ==========================================================================

-- ========================
-- CONFIGURATION
-- ========================
local NEAT_CONFIG = {
    POPULATION_SIZE     = 150,
    INPUT_COUNT         = 13,
    OUTPUT_COUNT        = 6,

    -- Mutation rates
    WEIGHT_MUTATE_RATE  = 0.8,      -- chance to mutate weights at all
    WEIGHT_PERTURB      = 0.1,      -- uniform perturbation range
    WEIGHT_RESET_RATE   = 0.1,      -- per-connection chance to fully reset
    ADD_NODE_RATE       = 0.03,
    ADD_CONNECTION_RATE = 0.05,
    DISABLE_RATE        = 0.04,

    -- Speciation coefficients
    EXCESS_COEFF        = 1.0,
    DISJOINT_COEFF      = 1.0,
    WEIGHT_COEFF        = 0.4,
    SPECIES_THRESHOLD   = 3.0,

    -- Stale species
    STALE_SPECIES_GENS  = 15,

    -- Evaluation
    FRAME_SKIP          = 4,
    TIMEOUT_FRAMES      = 300,      -- frames without X progress -> abort
    TARGET_X            = 3160,
    MAX_GENERATIONS     = 500,
    SAVESTATE_SLOT      = 1,
}

-- ========================
-- MEMORY ADDRESSES (matches genetic_bruteforce.lua / debug_overlay.lua)
-- ========================
local ADDR = {
    MARIO_X_PAGE    = 0x006D,
    MARIO_X_POS     = 0x0086,
    MARIO_Y         = 0x00CE,
    MARIO_VEL_X     = 0x0057,
    MARIO_VEL_Y     = 0x009F,
    LIVES           = 0x075A,
    TIME_H          = 0x07F8,
    TIME_T          = 0x07F9,
    TIME_O          = 0x07FA,
    FLOAT_STATE     = 0x001D,
    PLAYER_STATE    = 0x000E,
    CAMERA_X        = 0x03AD,

    -- Enemy slots (5 slots)
    ENEMY_X_POS     = 0x0087,   -- base; slots at +0..+4
    ENEMY_Y_POS     = 0x00CF,   -- base; slots at +0..+4

    -- Level tile layout (PPU nametable mirror in RAM)
    LEVEL_LAYOUT    = 0x0500,   -- 13 rows x 16 columns per page
}

-- ========================
-- HELPERS -- game state reading
-- ========================

local function get_mario_x()
    return memory.readbyte(ADDR.MARIO_X_PAGE) * 256 + memory.readbyte(ADDR.MARIO_X_POS)
end

local function get_mario_y()
    return memory.readbyte(ADDR.MARIO_Y)
end

local function is_dead()
    local state = memory.readbyte(ADDR.PLAYER_STATE)
    return state == 0x0B or state == 0x06
end

local function get_timer()
    return memory.readbyte(ADDR.TIME_H) * 100
         + memory.readbyte(ADDR.TIME_T) * 10
         + memory.readbyte(ADDR.TIME_O)
end

local function signed_byte(v)
    if v > 127 then return v - 256 end
    return v
end

-- Read a metatile from the level layout RAM.
-- world_x is Mario's absolute X; y_screen is screen-relative Y (0-239).
-- Returns the tile byte (0 = empty/air, >0 = solid-ish).
local function read_metatile(world_x, y_screen)
    local page   = math.floor(world_x / 256) % 2
    local sub_x  = math.floor((world_x % 256) / 16)
    local sub_y  = math.floor((y_screen - 32) / 16)
    if sub_y < 0 or sub_y > 12 or sub_x < 0 or sub_x > 15 then return 0 end
    local addr = ADDR.LEVEL_LAYOUT + page * (13 * 16) + sub_y * 16 + sub_x
    return memory.readbyte(addr)
end

-- ========================
-- INPUT VECTOR (13 neurons)
-- ========================

local function build_inputs()
    local inputs = {}
    local mario_x = get_mario_x()
    local mario_y = get_mario_y()
    local mario_screen_x = memory.readbyte(ADDR.CAMERA_X)  -- camera page for enemy delta

    -- 1. mario_x_vel (normalized roughly to [-1,1])
    local vx = signed_byte(memory.readbyte(ADDR.MARIO_VEL_X))
    inputs[1] = vx / 40.0

    -- 2. mario_y_vel
    local vy = signed_byte(memory.readbyte(ADDR.MARIO_VEL_Y))
    inputs[2] = vy / 40.0

    -- 3. on_ground (0x001D == 0 means grounded)
    inputs[3] = (memory.readbyte(ADDR.FLOAT_STATE) == 0) and 1.0 or 0.0

    -- 4. mario_y normalized
    inputs[4] = mario_y / 240.0

    -- 5-6. closest enemy dx, dy
    local closest_dx = 999
    local closest_dy = 0
    local closest_dist = 999999
    for slot = 0, 4 do
        local ex = memory.readbyte(ADDR.ENEMY_X_POS + slot)
        local ey = memory.readbyte(0x00CF + slot)
        -- Enemy screen X - mario screen X approximation
        local dx = ex - memory.readbyte(0x03AD)  -- camera-relative
        local dy = ey - mario_y
        local dist = dx * dx + dy * dy
        if dist < closest_dist and dist > 0 then
            closest_dist = dist
            closest_dx = dx
            closest_dy = dy
        end
    end
    inputs[5] = math.max(-1, math.min(1, closest_dx / 128.0))
    inputs[6] = math.max(-1, math.min(1, closest_dy / 128.0))

    -- 7. pit_ahead: sample 2-3 tiles ahead at ground level for gaps
    local pit_score = 0
    for look = 1, 3 do
        local check_x = mario_x + look * 16
        local ground_tile = read_metatile(check_x, 208)  -- near bottom of screen
        local below_tile  = read_metatile(check_x, 224)
        if ground_tile == 0 and below_tile == 0 then
            pit_score = pit_score + 1
        end
    end
    inputs[7] = pit_score / 3.0

    -- 8. tile_above (metatile above Mario's head)
    inputs[8] = (read_metatile(mario_x, mario_y - 16) > 0) and 1.0 or 0.0

    -- 9. tile_ahead_low (at Mario's feet, 1 block ahead)
    inputs[9] = (read_metatile(mario_x + 16, mario_y) > 0) and 1.0 or 0.0

    -- 10. tile_ahead_high (at Mario's head, 1 block ahead)
    inputs[10] = (read_metatile(mario_x + 16, mario_y - 16) > 0) and 1.0 or 0.0

    -- 11. tile_below (below Mario's feet)
    inputs[11] = (read_metatile(mario_x, mario_y + 16) > 0) and 1.0 or 0.0

    -- 12. time_remaining (normalized /400)
    inputs[12] = get_timer() / 400.0

    -- 13. x_progress (mario_x / 3168, 0-1)
    inputs[13] = mario_x / 3168.0

    return inputs
end

-- ========================
-- NEAT DATA STRUCTURES
-- ========================

-- Global innovation counter
local g_innovation = 0
-- Cache: maps "from:to" -> innovation number to reuse within a generation
local g_innovation_cache = {}

local function next_innovation(from_id, to_id)
    local key = from_id .. ":" .. to_id
    if g_innovation_cache[key] then
        return g_innovation_cache[key]
    end
    g_innovation = g_innovation + 1
    g_innovation_cache[key] = g_innovation
    return g_innovation
end

-- Reset innovation cache each generation (same structural mutation in same
-- generation gets the same number, different generations get new numbers)
local function reset_innovation_cache()
    g_innovation_cache = {}
end

-- Global node counter (IDs 1..INPUT_COUNT are inputs, INPUT_COUNT+1..INPUT_COUNT+OUTPUT_COUNT are outputs)
local g_next_node_id = NEAT_CONFIG.INPUT_COUNT + NEAT_CONFIG.OUTPUT_COUNT + 1

local function next_node_id()
    local id = g_next_node_id
    g_next_node_id = g_next_node_id + 1
    return id
end

-- Create a new empty genome
local function new_genome()
    return {
        fitness    = 0,
        adj_fitness = 0,
        max_x      = 0,
        genes      = {},   -- connection genes: {from, to, weight, enabled, innovation}
        nodes      = {},   -- set of node IDs present in this genome
        species    = 0,
    }
end

-- Initialize node set from genes + input/output nodes
local function rebuild_nodes(genome)
    local nodes = {}
    -- Inputs: 1..INPUT_COUNT
    for i = 1, NEAT_CONFIG.INPUT_COUNT do
        nodes[i] = "input"
    end
    -- Outputs: INPUT_COUNT+1 .. INPUT_COUNT+OUTPUT_COUNT
    for i = NEAT_CONFIG.INPUT_COUNT + 1, NEAT_CONFIG.INPUT_COUNT + NEAT_CONFIG.OUTPUT_COUNT do
        nodes[i] = "output"
    end
    -- Hidden: from genes
    for _, g in ipairs(genome.genes) do
        if not nodes[g.from] then nodes[g.from] = "hidden" end
        if not nodes[g.to]   then nodes[g.to]   = "hidden" end
    end
    genome.nodes = nodes
end

-- Create a minimal genome with direct input->output connections
local function create_basic_genome()
    local genome = new_genome()
    -- Connect each input to each output with small random weight
    for i = 1, NEAT_CONFIG.INPUT_COUNT do
        for j = 1, NEAT_CONFIG.OUTPUT_COUNT do
            local out_id = NEAT_CONFIG.INPUT_COUNT + j
            local gene = {
                from       = i,
                to         = out_id,
                weight     = (math.random() * 4 - 2),  -- [-2, 2]
                enabled    = true,
                innovation = next_innovation(i, out_id),
            }
            genome.genes[#genome.genes + 1] = gene
        end
    end
    rebuild_nodes(genome)
    return genome
end

-- Deep copy a genome
local function copy_genome(src)
    local dst = new_genome()
    dst.fitness     = src.fitness
    dst.adj_fitness = src.adj_fitness
    dst.max_x       = src.max_x
    dst.species     = src.species
    for _, g in ipairs(src.genes) do
        dst.genes[#dst.genes + 1] = {
            from       = g.from,
            to         = g.to,
            weight     = g.weight,
            enabled    = g.enabled,
            innovation = g.innovation,
        }
    end
    rebuild_nodes(dst)
    return dst
end

-- ========================
-- ACTIVATION FUNCTION
-- ========================

local function sigmoid(x)
    return 1.0 / (1.0 + math.exp(-4.9 * x))
end

-- ========================
-- FORWARD PROPAGATION
-- ========================

local function forward_propagate(genome, inputs)
    -- Build adjacency: incoming connections for each node
    local incoming = {}  -- node_id -> list of {from, weight}
    for _, g in ipairs(genome.genes) do
        if g.enabled then
            if not incoming[g.to] then incoming[g.to] = {} end
            incoming[g.to][#incoming[g.to] + 1] = { from = g.from, weight = g.weight }
        end
    end

    -- Activation values
    local activation = {}

    -- Set input activations
    for i = 1, NEAT_CONFIG.INPUT_COUNT do
        activation[i] = inputs[i] or 0
    end

    -- Topological evaluation: process nodes in dependency order.
    -- Since networks are feed-forward (no cycles by construction),
    -- we iteratively resolve nodes whose inputs are all computed.
    -- Use multiple passes to handle deep networks.
    local resolved = {}
    for i = 1, NEAT_CONFIG.INPUT_COUNT do
        resolved[i] = true
    end

    local max_passes = 20  -- safety limit for deep networks
    for pass = 1, max_passes do
        local progress = false
        for node_id, ntype in pairs(genome.nodes) do
            if not resolved[node_id] then
                -- Check if all inputs to this node are resolved
                local all_ready = true
                local inc = incoming[node_id]
                if inc then
                    for _, conn in ipairs(inc) do
                        if not resolved[conn.from] then
                            all_ready = false
                            break
                        end
                    end
                end
                if all_ready then
                    -- Compute activation
                    local sum = 0
                    if inc then
                        for _, conn in ipairs(inc) do
                            sum = sum + (activation[conn.from] or 0) * conn.weight
                        end
                    end
                    activation[node_id] = sigmoid(sum)
                    resolved[node_id] = true
                    progress = true
                end
            end
        end
        if not progress then break end
    end

    -- Ensure all output nodes have a value (default 0 if unresolved)
    for i = 1, NEAT_CONFIG.OUTPUT_COUNT do
        local out_id = NEAT_CONFIG.INPUT_COUNT + i
        if not activation[out_id] then
            activation[out_id] = 0
        end
    end

    -- Extract output values
    local outputs = {}
    for i = 1, NEAT_CONFIG.OUTPUT_COUNT do
        outputs[i] = activation[NEAT_CONFIG.INPUT_COUNT + i]
    end
    return outputs
end

-- ========================
-- OUTPUT -> JOYPAD MAPPING
-- ========================
-- Outputs: 1=right, 2=left, 3=jump(A), 4=run(B), 5=right+jump, 6=right+jump+run
-- Threshold: > 0.5 means active. Highest movement output wins.

local function outputs_to_joypad(outputs)
    local buttons = {}
    local threshold = 0.5

    -- Movement outputs (1=right, 2=left, 5=right+jump, 6=right+jump+run)
    -- Find highest-activated movement output
    local move_outputs = {
        { idx = 1, act = outputs[1] or 0, btns = { right = true } },
        { idx = 2, act = outputs[2] or 0, btns = { left = true } },
        { idx = 5, act = outputs[5] or 0, btns = { right = true, A = true } },
        { idx = 6, act = outputs[6] or 0, btns = { right = true, A = true, B = true } },
    }

    local best_move = nil
    local best_act  = threshold
    for _, m in ipairs(move_outputs) do
        if m.act > best_act then
            best_act  = m.act
            best_move = m.btns
        end
    end

    if best_move then
        for k, v in pairs(best_move) do buttons[k] = v end
    end

    -- Independent modifiers
    if (outputs[3] or 0) > threshold then buttons.A = true end  -- jump
    if (outputs[4] or 0) > threshold then buttons.B = true end  -- run

    return buttons
end

-- ========================
-- MUTATIONS
-- ========================

local function mutate_weights(genome)
    for _, g in ipairs(genome.genes) do
        if math.random() < NEAT_CONFIG.WEIGHT_RESET_RATE then
            g.weight = math.random() * 4 - 2  -- [-2, 2]
        else
            g.weight = g.weight + (math.random() * 2 - 1) * NEAT_CONFIG.WEIGHT_PERTURB
        end
    end
end

local function mutate_add_connection(genome)
    -- Collect all node IDs
    local node_ids = {}
    for id, _ in pairs(genome.nodes) do
        node_ids[#node_ids + 1] = id
    end
    if #node_ids < 2 then return end

    -- Try up to 50 times to find a valid new connection
    for _ = 1, 50 do
        local from_id = node_ids[math.random(#node_ids)]
        local to_id   = node_ids[math.random(#node_ids)]
        local valid = true

        -- No self-connections
        if from_id == to_id then valid = false end

        -- No connections TO input nodes
        if valid and genome.nodes[to_id] == "input" then valid = false end

        -- No connections FROM output nodes
        if valid and genome.nodes[from_id] == "output" then valid = false end

        -- Check if connection already exists
        if valid then
            for _, g in ipairs(genome.genes) do
                if g.from == from_id and g.to == to_id then
                    valid = false
                    break
                end
            end
        end

        -- Prevent cycles: for hidden->hidden, only allow lower ID -> higher ID
        if valid and genome.nodes[from_id] == "hidden" and genome.nodes[to_id] == "hidden" then
            if from_id >= to_id then valid = false end
        end

        -- Add the connection if all checks passed
        if valid then
            genome.genes[#genome.genes + 1] = {
                from       = from_id,
                to         = to_id,
                weight     = math.random() * 4 - 2,
                enabled    = true,
                innovation = next_innovation(from_id, to_id),
            }
            return
        end
    end
end

local function mutate_add_node(genome)
    -- Pick a random enabled connection to split
    local enabled_genes = {}
    for i, g in ipairs(genome.genes) do
        if g.enabled then
            enabled_genes[#enabled_genes + 1] = i
        end
    end
    if #enabled_genes == 0 then return end

    local idx  = enabled_genes[math.random(#enabled_genes)]
    local gene = genome.genes[idx]

    -- Disable the old connection
    gene.enabled = false

    -- Create new hidden node
    local new_id = next_node_id()
    genome.nodes[new_id] = "hidden"

    -- Connection 1: from -> new_node (weight 1.0 to preserve behavior)
    genome.genes[#genome.genes + 1] = {
        from       = gene.from,
        to         = new_id,
        weight     = 1.0,
        enabled    = true,
        innovation = next_innovation(gene.from, new_id),
    }

    -- Connection 2: new_node -> to (original weight)
    genome.genes[#genome.genes + 1] = {
        from       = new_id,
        to         = gene.to,
        weight     = gene.weight,
        enabled    = true,
        innovation = next_innovation(new_id, gene.to),
    }
end

local function mutate_disable(genome)
    local enabled_genes = {}
    for i, g in ipairs(genome.genes) do
        if g.enabled then
            enabled_genes[#enabled_genes + 1] = i
        end
    end
    if #enabled_genes == 0 then return end
    local idx = enabled_genes[math.random(#enabled_genes)]
    genome.genes[idx].enabled = false
end

local function mutate(genome)
    -- Weight mutations
    if math.random() < NEAT_CONFIG.WEIGHT_MUTATE_RATE then
        mutate_weights(genome)
    end

    -- Structural mutations
    if math.random() < NEAT_CONFIG.ADD_CONNECTION_RATE then
        mutate_add_connection(genome)
    end
    if math.random() < NEAT_CONFIG.ADD_NODE_RATE then
        mutate_add_node(genome)
    end
    if math.random() < NEAT_CONFIG.DISABLE_RATE then
        mutate_disable(genome)
    end

    rebuild_nodes(genome)
end

-- ========================
-- CROSSOVER
-- ========================

local function crossover(parent1, parent2)
    -- parent1 should be the fitter parent
    if parent2.fitness > parent1.fitness then
        parent1, parent2 = parent2, parent1
    end

    local child = new_genome()

    -- Index parent2 genes by innovation number
    local p2_by_innov = {}
    for _, g in ipairs(parent2.genes) do
        p2_by_innov[g.innovation] = g
    end

    for _, g1 in ipairs(parent1.genes) do
        local g2 = p2_by_innov[g1.innovation]
        if g2 then
            -- Matching gene: pick randomly from either parent
            local src = (math.random() < 0.5) and g1 or g2
            child.genes[#child.genes + 1] = {
                from       = src.from,
                to         = src.to,
                weight     = src.weight,
                enabled    = src.enabled,
                innovation = src.innovation,
            }
            -- If either parent has it disabled, 75% chance child gets it disabled
            if not g1.enabled or not g2.enabled then
                if math.random() < 0.75 then
                    child.genes[#child.genes].enabled = false
                end
            end
        else
            -- Excess/disjoint gene from fitter parent -> include
            child.genes[#child.genes + 1] = {
                from       = g1.from,
                to         = g1.to,
                weight     = g1.weight,
                enabled    = g1.enabled,
                innovation = g1.innovation,
            }
        end
    end

    rebuild_nodes(child)
    return child
end

-- ========================
-- SPECIATION
-- ========================

local function compatibility_distance(genome1, genome2)
    -- Sort genes by innovation number for alignment
    local g1 = {}
    local g2 = {}
    for _, g in ipairs(genome1.genes) do g1[#g1 + 1] = g end
    for _, g in ipairs(genome2.genes) do g2[#g2 + 1] = g end
    table.sort(g1, function(a, b) return a.innovation < b.innovation end)
    table.sort(g2, function(a, b) return a.innovation < b.innovation end)

    local i, j = 1, 1
    local excess   = 0
    local disjoint = 0
    local weight_diff_sum = 0
    local matching = 0

    while i <= #g1 and j <= #g2 do
        if g1[i].innovation == g2[j].innovation then
            -- Matching
            matching = matching + 1
            weight_diff_sum = weight_diff_sum + math.abs(g1[i].weight - g2[j].weight)
            i = i + 1
            j = j + 1
        elseif g1[i].innovation < g2[j].innovation then
            disjoint = disjoint + 1
            i = i + 1
        else
            disjoint = disjoint + 1
            j = j + 1
        end
    end

    -- Remaining genes are excess
    excess = (#g1 - i + 1) + (#g2 - j + 1)
    if i > #g1 then excess = #g2 - j + 1 end
    if j > #g2 then excess = #g1 - i + 1 end

    local N = math.max(#g1, #g2)
    if N < 20 then N = 1 end  -- small genomes: don't normalize

    local avg_w = 0
    if matching > 0 then avg_w = weight_diff_sum / matching end

    return NEAT_CONFIG.EXCESS_COEFF * excess / N
         + NEAT_CONFIG.DISJOINT_COEFF * disjoint / N
         + NEAT_CONFIG.WEIGHT_COEFF * avg_w
end

-- Species data structure
-- { id, genomes={}, best_fitness, staleness, representative }
local g_species_list = {}
local g_next_species_id = 1

local function assign_species(population)
    -- Preserve representatives from existing species
    local old_reps = {}
    for _, sp in ipairs(g_species_list) do
        if sp.representative then
            old_reps[sp.id] = sp.representative
        end
    end

    -- Clear genomes from all species but keep metadata
    for _, sp in ipairs(g_species_list) do
        sp.genomes = {}
    end

    for _, genome in ipairs(population) do
        local placed = false
        for _, sp in ipairs(g_species_list) do
            if sp.representative then
                local dist = compatibility_distance(genome, sp.representative)
                if dist < NEAT_CONFIG.SPECIES_THRESHOLD then
                    sp.genomes[#sp.genomes + 1] = genome
                    genome.species = sp.id
                    placed = true
                    break
                end
            end
        end
        if not placed then
            -- New species
            local sp = {
                id             = g_next_species_id,
                genomes        = { genome },
                best_fitness   = 0,
                staleness      = 0,
                representative = copy_genome(genome),
            }
            g_next_species_id = g_next_species_id + 1
            g_species_list[#g_species_list + 1] = sp
            genome.species = sp.id
        end
    end

    -- Remove empty species
    local alive = {}
    for _, sp in ipairs(g_species_list) do
        if #sp.genomes > 0 then
            -- Update representative to a random member
            sp.representative = copy_genome(sp.genomes[math.random(#sp.genomes)])
            alive[#alive + 1] = sp
        end
    end
    g_species_list = alive
end

-- ========================
-- FITNESS SHARING & SELECTION
-- ========================

local function adjust_fitness()
    for _, sp in ipairs(g_species_list) do
        local n = #sp.genomes
        for _, genome in ipairs(sp.genomes) do
            genome.adj_fitness = genome.fitness / n
        end
    end
end

local function update_staleness()
    for _, sp in ipairs(g_species_list) do
        -- Find best fitness in species
        local best = 0
        for _, genome in ipairs(sp.genomes) do
            if genome.fitness > best then best = genome.fitness end
        end
        if best > sp.best_fitness then
            sp.best_fitness = best
            sp.staleness = 0
        else
            sp.staleness = sp.staleness + 1
        end
    end
end

-- Kill stale species (keep only champion)
local function cull_stale_species()
    for _, sp in ipairs(g_species_list) do
        if sp.staleness >= NEAT_CONFIG.STALE_SPECIES_GENS and #sp.genomes > 1 then
            -- Sort by fitness descending
            table.sort(sp.genomes, function(a, b) return a.fitness > b.fitness end)
            -- Keep only the champion
            local champ = sp.genomes[1]
            sp.genomes = { champ }
        end
    end
end

-- Within each species, keep top 50%
local function cull_species_bottom()
    for _, sp in ipairs(g_species_list) do
        table.sort(sp.genomes, function(a, b) return a.fitness > b.fitness end)
        local keep = math.max(1, math.floor(#sp.genomes / 2))
        local new_genomes = {}
        for i = 1, keep do
            new_genomes[i] = sp.genomes[i]
        end
        sp.genomes = new_genomes
    end
end

-- Select a parent from a species using tournament selection
local function select_parent(species)
    if #species.genomes == 1 then return species.genomes[1] end
    -- Simple tournament of size 2
    local a = species.genomes[math.random(#species.genomes)]
    local b = species.genomes[math.random(#species.genomes)]
    if a.fitness >= b.fitness then return a else return b end
end

-- ========================
-- BREED NEXT GENERATION
-- ========================

local function breed_next_generation()
    -- Calculate total adjusted fitness for proportional offspring allocation
    local total_adj = 0
    for _, sp in ipairs(g_species_list) do
        for _, g in ipairs(sp.genomes) do
            total_adj = total_adj + g.adj_fitness
        end
    end

    local new_population = {}

    -- Elitism: champion of each species with > 5 members is carried over unchanged
    for _, sp in ipairs(g_species_list) do
        if #sp.genomes >= 5 then
            table.sort(sp.genomes, function(a, b) return a.fitness > b.fitness end)
            new_population[#new_population + 1] = copy_genome(sp.genomes[1])
        end
    end

    -- Allocate offspring proportional to species adjusted fitness
    local species_offspring = {}
    local remaining = NEAT_CONFIG.POPULATION_SIZE - #new_population
    for _, sp in ipairs(g_species_list) do
        local sp_adj = 0
        for _, g in ipairs(sp.genomes) do
            sp_adj = sp_adj + g.adj_fitness
        end
        local alloc = 0
        if total_adj > 0 then
            alloc = math.floor((sp_adj / total_adj) * remaining)
        end
        species_offspring[sp.id] = alloc
    end

    -- Distribute any rounding remainder to the best species
    local allocated = 0
    for _, v in pairs(species_offspring) do allocated = allocated + v end
    local deficit = remaining - allocated
    if deficit > 0 and #g_species_list > 0 then
        -- Find best species
        local best_sp = g_species_list[1]
        for _, sp in ipairs(g_species_list) do
            if sp.best_fitness > best_sp.best_fitness then best_sp = sp end
        end
        species_offspring[best_sp.id] = (species_offspring[best_sp.id] or 0) + deficit
    end

    -- Breed offspring within each species
    for _, sp in ipairs(g_species_list) do
        local count = species_offspring[sp.id] or 0
        for _ = 1, count do
            local child
            if #sp.genomes == 1 then
                child = copy_genome(sp.genomes[1])
                mutate(child)
            elseif math.random() < 0.75 then
                -- Crossover
                local p1 = select_parent(sp)
                local p2 = select_parent(sp)
                child = crossover(p1, p2)
                mutate(child)
            else
                -- Clone + mutate
                local p = select_parent(sp)
                child = copy_genome(p)
                mutate(child)
            end
            new_population[#new_population + 1] = child
        end
    end

    -- If population is under target (due to rounding), fill with mutated copies of best
    while #new_population < NEAT_CONFIG.POPULATION_SIZE do
        if #g_species_list > 0 and #g_species_list[1].genomes > 0 then
            local filler = copy_genome(g_species_list[1].genomes[1])
            mutate(filler)
            new_population[#new_population + 1] = filler
        else
            new_population[#new_population + 1] = create_basic_genome()
        end
    end

    -- Trim if over
    while #new_population > NEAT_CONFIG.POPULATION_SIZE do
        new_population[#new_population] = nil
    end

    return new_population
end

-- ========================
-- EVALUATE A GENOME
-- ========================

local function evaluate_genome(genome, show_overlay)
    if show_overlay then
        emu.speedmode("normal")
    else
        emu.speedmode("turbo")
    end

    -- Load save state
    savestate.load(savestate.object(NEAT_CONFIG.SAVESTATE_SLOT))

    -- Settling delay
    for i = 1, 30 do emu.frameadvance() end

    local max_x = 0
    local frames_stuck = 0
    local total_frames = 0
    local completed = false

    -- Main evaluation loop
    local max_eval_steps = 3000  -- safety limit (~12000 frames at skip=4)
    for step = 1, max_eval_steps do
        -- Read inputs
        local inputs = build_inputs()

        -- Forward propagate
        local outputs = forward_propagate(genome, inputs)

        -- Convert to joypad
        local buttons = outputs_to_joypad(outputs)

        -- Execute for FRAME_SKIP frames
        for f = 1, NEAT_CONFIG.FRAME_SKIP do
            joypad.set(1, buttons)

            -- Draw overlay on last frame if requested
            if show_overlay and f == NEAT_CONFIG.FRAME_SKIP then
                draw_network_overlay(genome, inputs, outputs)
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
            frames_stuck = frames_stuck + NEAT_CONFIG.FRAME_SKIP
        end

        -- Termination conditions
        if is_dead() then break end

        if current_x >= NEAT_CONFIG.TARGET_X then
            completed = true
            break
        end

        if frames_stuck > NEAT_CONFIG.TIMEOUT_FRAMES then break end

        if get_timer() <= 0 then break end
    end

    -- Calculate fitness
    local fitness = max_x
    if completed then
        fitness = fitness + 10000 + (400 - total_frames / 60) * 10
    end

    genome.fitness = fitness
    genome.max_x   = max_x

    return fitness, max_x, completed, total_frames
end

-- ========================
-- NETWORK VISUALIZATION OVERLAY
-- ========================

-- Forward declaration for evaluate_genome to reference
function draw_network_overlay(genome, inputs, outputs)
    -- Background panel
    gui.drawbox(0, 0, 130, 80, "#000000C0", "#000000C0")

    local x = get_mario_x()
    local progress = math.min(100, x / 3168 * 100)
    gui.text(2, 2, string.format("NEAT X:%d (%.0f%%)", x, progress), "#44FF44")
    gui.text(2, 12, string.format("Nodes:%d Genes:%d", count_table(genome.nodes), #genome.genes), "#FFFFFF")

    -- Show output decisions
    local out_names = {"R", "L", "A", "B", "R+A", "R+A+B"}
    local out_str = ""
    for i = 1, NEAT_CONFIG.OUTPUT_COUNT do
        if (outputs[i] or 0) > 0.5 then
            out_str = out_str .. out_names[i] .. " "
        end
    end
    gui.text(2, 22, "Out: " .. out_str, "#FFFF44")

    -- Show top input values
    local in_names = {"vX","vY","gnd","mY","eDX","eDY","pit","tUp","tAL","tAH","tBl","tim","prg"}
    gui.text(2, 32, string.format("vX:%.1f vY:%.1f gnd:%d pit:%.1f",
        inputs[1] or 0, inputs[2] or 0, inputs[3] or 0, inputs[7] or 0), "#AAAAAA")
    gui.text(2, 42, string.format("eDX:%.1f eDY:%.1f prg:%.2f",
        inputs[5] or 0, inputs[6] or 0, inputs[13] or 0), "#AAAAAA")

    -- Mini network visualization (right side of screen)
    draw_network_graph(genome, inputs, outputs)
end

function count_table(t)
    local n = 0
    for _ in pairs(t) do n = n + 1 end
    return n
end

function draw_network_graph(genome, inputs, outputs)
    -- Draw a compact network graph on the right side
    local base_x = 160
    local base_y = 100
    local layer_w = 30
    local node_r  = 2

    -- Categorize nodes by type
    local input_nodes  = {}
    local hidden_nodes = {}
    local output_nodes = {}
    for id, ntype in pairs(genome.nodes) do
        if ntype == "input"  then input_nodes[#input_nodes + 1]   = id
        elseif ntype == "hidden" then hidden_nodes[#hidden_nodes + 1] = id
        elseif ntype == "output" then output_nodes[#output_nodes + 1] = id
        end
    end
    table.sort(input_nodes)
    table.sort(hidden_nodes)
    table.sort(output_nodes)

    -- Position nodes
    local positions = {}  -- id -> {x, y}

    -- Input column
    local in_spacing = math.min(10, 120 / math.max(1, #input_nodes))
    for i, id in ipairs(input_nodes) do
        positions[id] = { x = base_x, y = base_y + (i - 1) * in_spacing }
    end

    -- Hidden column(s) -- stack vertically in middle
    local h_spacing = math.min(10, 120 / math.max(1, #hidden_nodes))
    for i, id in ipairs(hidden_nodes) do
        positions[id] = { x = base_x + layer_w, y = base_y + (i - 1) * h_spacing }
    end

    -- Output column
    local out_spacing = math.min(15, 120 / math.max(1, #output_nodes))
    for i, id in ipairs(output_nodes) do
        positions[id] = { x = base_x + layer_w * 2, y = base_y + (i - 1) * out_spacing }
    end

    -- Draw connections (enabled only)
    for _, g in ipairs(genome.genes) do
        if g.enabled and positions[g.from] and positions[g.to] then
            local p1 = positions[g.from]
            local p2 = positions[g.to]
            -- Color by weight: positive = green, negative = red
            -- Brightness by activation flow (approximate with weight magnitude)
            local mag = math.min(1, math.abs(g.weight) / 2)
            local alpha = math.floor(80 + 175 * mag)
            local color
            if g.weight >= 0 then
                color = string.format("#00%02X00%02X", math.floor(128 + 127 * mag), alpha)
            else
                color = string.format("#%02X0000%02X", math.floor(128 + 127 * mag), alpha)
            end
            gui.drawline(p1.x, p1.y, p2.x, p2.y, color)
        end
    end

    -- Draw nodes
    for id, pos in pairs(positions) do
        local ntype = genome.nodes[id]
        local color = "#888888"
        if ntype == "input" then
            color = "#4488FF"
        elseif ntype == "output" then
            -- Color by activation
            local out_idx = id - NEAT_CONFIG.INPUT_COUNT
            local act = outputs[out_idx] or 0
            if act > 0.5 then
                color = "#44FF44"
            else
                color = "#FF8844"
            end
        elseif ntype == "hidden" then
            color = "#FFFF44"
        end
        gui.drawbox(pos.x - node_r, pos.y - node_r, pos.x + node_r, pos.y + node_r, color, color)
    end
end

-- ========================
-- SAVE / LOAD
-- ========================

local function serialize_genome(genome)
    local parts = {}
    parts[#parts + 1] = "return {"
    parts[#parts + 1] = string.format("  fitness = %d,", genome.fitness)
    parts[#parts + 1] = string.format("  max_x = %d,", genome.max_x)
    parts[#parts + 1] = "  genes = {"
    for _, g in ipairs(genome.genes) do
        parts[#parts + 1] = string.format(
            "    {from=%d, to=%d, weight=%.6f, enabled=%s, innovation=%d},",
            g.from, g.to, g.weight, tostring(g.enabled), g.innovation)
    end
    parts[#parts + 1] = "  },"

    -- Save node info
    parts[#parts + 1] = "  nodes = {"
    for id, ntype in pairs(genome.nodes) do
        parts[#parts + 1] = string.format('    [%d] = "%s",', id, ntype)
    end
    parts[#parts + 1] = "  },"
    parts[#parts + 1] = "}"
    return table.concat(parts, "\n")
end

local function save_best_genome(filename, genome, gen)
    local f = io.open(filename, "w")
    if not f then
        print("  [!] Failed to save to " .. filename)
        return
    end
    f:write("-- NEAT best genome for SMB 1-1\n")
    f:write(string.format("-- Generation: %d, Fitness: %d, Max X: %d\n", gen, genome.fitness, genome.max_x))
    f:write(string.format("-- Genes: %d, Nodes: %d\n", #genome.genes, count_table(genome.nodes)))
    f:write(serialize_genome(genome))
    f:write("\n")
    f:close()
    print(string.format("  Saved best genome to %s (fitness=%d, x=%d)", filename, genome.fitness, genome.max_x))
end

local function load_genome(filename)
    local f = io.open(filename, "r")
    if not f then return nil end
    local content = f:read("*all")
    f:close()
    local loader = loadstring(content)
    if not loader then return nil end
    local ok, data = pcall(loader)
    if not ok or not data then return nil end

    local genome = new_genome()
    genome.fitness = data.fitness or 0
    genome.max_x   = data.max_x or 0

    if data.genes then
        for _, g in ipairs(data.genes) do
            genome.genes[#genome.genes + 1] = {
                from       = g.from,
                to         = g.to,
                weight     = g.weight,
                enabled    = g.enabled,
                innovation = g.innovation,
            }
            -- Track highest innovation number
            if g.innovation > g_innovation then
                g_innovation = g.innovation
            end
        end
    end

    if data.nodes then
        genome.nodes = data.nodes
        -- Track highest node ID
        for id, _ in pairs(data.nodes) do
            if id >= g_next_node_id then
                g_next_node_id = id + 1
            end
        end
    else
        rebuild_nodes(genome)
    end

    print(string.format("  Loaded genome from %s (fitness=%d, x=%d, genes=%d)",
        filename, genome.fitness, genome.max_x, #genome.genes))
    return genome
end

local function save_population(filename, population, gen)
    local f = io.open(filename, "w")
    if not f then
        print("  [!] Failed to save population to " .. filename)
        return
    end
    f:write("-- NEAT population snapshot\n")
    f:write(string.format("-- Generation: %d, Size: %d\n", gen, #population))
    f:write(string.format("-- Innovation counter: %d, Next node ID: %d\n", g_innovation, g_next_node_id))
    f:write("return {\n")
    f:write(string.format("  generation = %d,\n", gen))
    f:write(string.format("  innovation = %d,\n", g_innovation))
    f:write(string.format("  next_node_id = %d,\n", g_next_node_id))
    f:write("  genomes = {\n")
    for idx, genome in ipairs(population) do
        f:write("    {\n")
        f:write(string.format("      fitness = %d,\n", genome.fitness))
        f:write(string.format("      max_x = %d,\n", genome.max_x))
        f:write("      genes = {\n")
        for _, g in ipairs(genome.genes) do
            f:write(string.format(
                "        {from=%d, to=%d, weight=%.6f, enabled=%s, innovation=%d},\n",
                g.from, g.to, g.weight, tostring(g.enabled), g.innovation))
        end
        f:write("      },\n")
        f:write("    },\n")
    end
    f:write("  },\n")
    f:write("}\n")
    f:close()
    print(string.format("  Saved population (%d genomes) to %s", #population, filename))
end

local function load_population(filename)
    local f = io.open(filename, "r")
    if not f then return nil end
    local content = f:read("*all")
    f:close()
    local loader = loadstring(content)
    if not loader then return nil end
    local ok, data = pcall(loader)
    if not ok or not data then return nil end

    local population = {}
    if data.innovation then
        g_innovation = math.max(g_innovation, data.innovation)
    end
    if data.next_node_id then
        g_next_node_id = math.max(g_next_node_id, data.next_node_id)
    end

    if data.genomes then
        for _, gdata in ipairs(data.genomes) do
            local genome = new_genome()
            genome.fitness = gdata.fitness or 0
            genome.max_x   = gdata.max_x or 0
            if gdata.genes then
                for _, g in ipairs(gdata.genes) do
                    genome.genes[#genome.genes + 1] = {
                        from       = g.from,
                        to         = g.to,
                        weight     = g.weight,
                        enabled    = g.enabled,
                        innovation = g.innovation,
                    }
                    if g.innovation > g_innovation then
                        g_innovation = g.innovation
                    end
                end
            end
            rebuild_nodes(genome)
            population[#population + 1] = genome
        end
    end

    print(string.format("  Loaded population: %d genomes from %s (gen=%s)",
        #population, filename, tostring(data.generation or "?")))
    return population, data.generation or 0
end

-- ========================
-- MAIN NEAT LOOP
-- ========================

local function run_neat()
    math.randomseed(os.time())

    print("=== NEAT Evolution - SMB 1-1 ===")
    print(string.format("Population: %d, Inputs: %d, Outputs: %d",
        NEAT_CONFIG.POPULATION_SIZE, NEAT_CONFIG.INPUT_COUNT, NEAT_CONFIG.OUTPUT_COUNT))
    print(string.format("Species threshold: %.1f, Max generations: %d",
        NEAT_CONFIG.SPECIES_THRESHOLD, NEAT_CONFIG.MAX_GENERATIONS))

    -- Initialize population
    local population = {}
    local start_gen  = 0

    -- Try to load existing population
    local loaded_pop, loaded_gen = load_population("neat_population.txt")
    if loaded_pop and #loaded_pop > 0 then
        population = loaded_pop
        start_gen  = loaded_gen
        print(string.format("  Resuming from generation %d with %d genomes", start_gen, #population))
    end

    -- Try to seed from saved best genome
    local seed_genome = load_genome("neat_best.txt")
    if seed_genome then
        if #population == 0 then
            -- Seed population with mutated copies
            population[1] = seed_genome
            for i = 2, math.min(20, NEAT_CONFIG.POPULATION_SIZE) do
                local variant = copy_genome(seed_genome)
                mutate(variant)
                population[i] = variant
            end
        else
            -- Inject as first member
            population[1] = seed_genome
        end
    end

    -- Fill remaining slots with basic genomes
    reset_innovation_cache()
    while #population < NEAT_CONFIG.POPULATION_SIZE do
        population[#population + 1] = create_basic_genome()
    end

    -- Trim if over
    while #population > NEAT_CONFIG.POPULATION_SIZE do
        population[#population] = nil
    end

    local all_time_best_fitness = 0
    local all_time_best_genome  = nil
    local all_time_best_x       = 0
    local level_completed       = false

    -- ========================
    -- GENERATION LOOP
    -- ========================
    for gen = start_gen + 1, NEAT_CONFIG.MAX_GENERATIONS do
        reset_innovation_cache()

        -- Evaluate each genome
        local gen_best_fitness = 0
        local gen_best_x       = 0
        local gen_best_idx     = 1

        for i = 1, #population do
            local fitness, max_x, completed, frames = evaluate_genome(population[i], false)

            if fitness > gen_best_fitness then
                gen_best_fitness = fitness
                gen_best_x       = max_x
                gen_best_idx     = i
            end

            if completed then
                level_completed = true
                print(string.format("!!! LEVEL COMPLETED !!! Gen %d, Genome %d, X=%d, Frames=%d",
                    gen, i, max_x, frames))
            end

            -- Progress ticker every 25 genomes
            if i % 25 == 0 then
                gui.drawbox(0, 224, 255, 240, "#000000", "#000000")
                gui.text(2, 226, string.format("Gen %d: Eval %d/%d  Best X: %d",
                    gen, i, #population, gen_best_x), "#FFFF44")
                emu.frameadvance()
            end
        end

        -- Update all-time best
        if gen_best_fitness > all_time_best_fitness then
            all_time_best_fitness = gen_best_fitness
            all_time_best_x       = gen_best_x
            all_time_best_genome  = copy_genome(population[gen_best_idx])

            -- Save new best
            save_best_genome("neat_best.txt", all_time_best_genome, gen)
            print(string.format("  >>> NEW BEST: fitness=%d, x=%d (%.1f%%)",
                all_time_best_fitness, all_time_best_x, all_time_best_x / 3168 * 100))
        end

        -- Speciation
        assign_species(population)

        -- Fitness sharing
        adjust_fitness()

        -- Staleness tracking
        update_staleness()

        -- Generation summary
        local avg_fitness = 0
        for _, g in ipairs(population) do avg_fitness = avg_fitness + g.fitness end
        avg_fitness = avg_fitness / #population

        -- Find best species
        local best_sp_id = 0
        local best_sp_fit = 0
        for _, sp in ipairs(g_species_list) do
            if sp.best_fitness > best_sp_fit then
                best_sp_fit = sp.best_fitness
                best_sp_id  = sp.id
            end
        end

        print(string.format("Gen %d: species=%d, best_x=%d, avg=%.0f, best_species=#%d, all_time=%d (%.1f%%)",
            gen, #g_species_list, gen_best_x, avg_fitness, best_sp_id,
            all_time_best_x, all_time_best_x / 3168 * 100))

        -- Species breakdown every 10 generations
        if gen % 10 == 0 then
            print("  Species breakdown:")
            for _, sp in ipairs(g_species_list) do
                print(string.format("    #%d: size=%d, best=%.0f, stale=%d",
                    sp.id, #sp.genomes, sp.best_fitness, sp.staleness))
            end
        end

        -- Save population every 10 generations
        if gen % 10 == 0 then
            save_population("neat_population.txt", population, gen)
        end

        -- Replay best genome every 10 generations
        if gen % 10 == 0 and all_time_best_genome then
            print(string.format("  Replaying best (x=%d)...", all_time_best_x))
            evaluate_genome(all_time_best_genome, true)
        end

        -- Stop if level completed
        if level_completed then
            save_best_genome("neat_best.txt", all_time_best_genome, gen)
            save_population("neat_population.txt", population, gen)
            break
        end

        -- ========================
        -- BREED NEXT GENERATION
        -- ========================

        -- Cull stale species
        cull_stale_species()

        -- Within each species, keep top 50%
        cull_species_bottom()

        -- Breed
        population = breed_next_generation()
    end

    -- ========================
    -- FINAL RESULTS
    -- ========================
    if all_time_best_genome then
        print("\n=== NEAT EVOLUTION COMPLETE ===")
        print(string.format("Best fitness: %d, Max X: %d (%.1f%%)",
            all_time_best_fitness, all_time_best_x, all_time_best_x / 3168 * 100))
        print(string.format("Genes: %d, Nodes: %d",
            #all_time_best_genome.genes, count_table(all_time_best_genome.nodes)))

        if level_completed then
            print(">>> LEVEL COMPLETED! <<<")
        end

        -- Final replay at normal speed
        print("Final replay...")
        emu.speedmode("normal")
        evaluate_genome(all_time_best_genome, true)

        -- Save final state
        save_best_genome("neat_best.txt", all_time_best_genome, NEAT_CONFIG.MAX_GENERATIONS)
        save_population("neat_population.txt", {all_time_best_genome}, NEAT_CONFIG.MAX_GENERATIONS)
    end

    return all_time_best_genome
end

-- ========================
-- START
-- ========================
print("=== NEAT NeuroEvolution for SMB 1-1 ===")
print("Evolves neural network topologies from scratch.")
print("Make sure save state slot 1 has World 1-1 start!")
print("Starting in 2 seconds...")

-- Brief delay so the user sees the message
for i = 1, 120 do emu.frameadvance() end

run_neat()

print("\n=== NEAT COMPLETE ===")
