-- ==========================================================================
-- Winning Genome Analyzer - Map actions to positions and obstacles
-- ==========================================================================
-- Replays a winning genome from genetic_bruteforce.lua and logs:
-- - Every action taken at every X position
-- - Jump timing relative to pits and enemies
-- - Airborne state and height profile
-- - Where the agent slows down, speeds up, or reverses
--
-- Usage: Make sure winning_genome.txt exists, save state slot 1 = 1-1 start,
--        then load this script in FCEUX.
--
-- Output: genome_analysis.csv (per-decision-frame data)
--         genome_analysis_summary.txt (human-readable obstacle report)
-- ==========================================================================

-- ========================
-- CONFIG
-- ========================
local FRAME_SKIP = 4
local SAVESTATE_SLOT = 1
local SETTLE_FRAMES = 30  -- Must match genetic_bruteforce.lua CONFIG.SETTLE_FRAMES
                          -- Set to 0 if the genome was evolved without settling delay

-- ========================
-- ACTION MAPPING
-- ========================
local ACTIONS = {
    [0]  = {},
    [1]  = {right = true},
    [2]  = {left = true},
    [3]  = {A = true},
    [4]  = {right = true, A = true},
    [5]  = {left = true, A = true},
    [6]  = {B = true},
    [7]  = {right = true, B = true},
    [8]  = {left = true, B = true},
    [9]  = {right = true, A = true, B = true},
    [10] = {left = true, A = true, B = true},
    [11] = {down = true},
}

local ACTION_NAMES = {
    [0]  = "NOOP",
    [1]  = "Right",
    [2]  = "Left",
    [3]  = "Jump",
    [4]  = "Right+Jump",
    [5]  = "Left+Jump",
    [6]  = "Run",
    [7]  = "Right+Run",
    [8]  = "Left+Run",
    [9]  = "Right+Jump+Run",
    [10] = "Left+Jump+Run",
    [11] = "Down",
}

-- ========================
-- MEMORY ADDRESSES
-- ========================
local ADDR = {
    MARIO_X_PAGE = 0x006D,
    MARIO_X_POS  = 0x0086,
    MARIO_Y      = 0x00CE,
    FLOAT_STATE  = 0x001D,
    VEL_X        = 0x0057,
    VEL_Y        = 0x009F,
    PLAYER_STATE = 0x000E,
    LIVES        = 0x075A,
    POWER        = 0x0756,
    TIME_H       = 0x07F8,
}

local function get_mario_x()
    return memory.readbyte(ADDR.MARIO_X_PAGE) * 256 + memory.readbyte(ADDR.MARIO_X_POS)
end

local function signed_byte(v)
    return v > 127 and (v - 256) or v
end

local function is_dead()
    local s = memory.readbyte(ADDR.PLAYER_STATE)
    return s == 0x0B or s == 0x06
end

-- ========================
-- LOAD GENOME
-- ========================
local function load_genome()
    -- Try loading from winning_genome.txt as a Lua table
    local f = io.open("winning_genome.txt", "r")
    if not f then
        print("ERROR: winning_genome.txt not found!")
        print("Run genetic_bruteforce.lua first to generate a winning genome.")
        return nil
    end
    
    local content = f:read("*all")
    f:close()
    
    -- The file format is: return {1,9,4,7,...}
    -- Load it as a Lua chunk
    local loader = loadstring(content)
    if not loader then
        print("ERROR: Could not parse winning_genome.txt")
        return nil
    end
    
    local genome = loader()
    if not genome or #genome == 0 then
        print("ERROR: Genome is empty")
        return nil
    end
    
    print(string.format("Loaded genome with %d actions", #genome))
    return genome
end

-- ========================
-- REPLAY AND ANALYZE
-- ========================
local function analyze_genome(genome)
    -- Open CSV output
    local csv = io.open("genome_analysis.csv", "w")
    csv:write("gene_idx,frame,action_id,action_name,mario_x,mario_y,vel_x,vel_y,")
    csv:write("float_state,on_ground,power,delta_x,delta_y,is_jumping,is_airborne\n")
    
    -- Load save state
    savestate.load(savestate.object(SAVESTATE_SLOT))
    emu.speedmode("normal")
    
    -- Settling delay: must match the GA's delay when the genome was evolved.
    -- If the genome was evolved with SETTLE_FRAMES=0, set this to 0 too.
    if SETTLE_FRAMES > 0 then
        for i = 1, SETTLE_FRAMES do emu.frameadvance() end
    end
    
    -- Tracking
    local prev_x = get_mario_x()
    local prev_y = memory.readbyte(ADDR.MARIO_Y)
    local max_x = 0
    local total_frames = 0
    local jump_start_x = nil
    local jump_start_frame = nil
    local jump_peak_y = 999
    
    -- Obstacle log entries
    local obstacles = {}
    local jumps = {}
    local action_at_x = {}  -- x -> {action, count}
    
    print("=== Replaying and analyzing genome ===")
    
    for gene_idx = 1, #genome do
        local action_id = genome[gene_idx]
        local action = ACTIONS[action_id] or {}
        local action_name = ACTION_NAMES[action_id] or "?"
        
        -- Execute action for FRAME_SKIP frames
        for f = 1, FRAME_SKIP do
            joypad.set(1, action)
            
            -- Draw overlay
            local x = get_mario_x()
            local y = memory.readbyte(ADDR.MARIO_Y)
            local progress = math.min(100, x / 3168 * 100)
            gui.drawbox(0, 0, 140, 32, "#000000C0", "#000000C0")
            gui.text(2, 2,  string.format("X:%d (%.1f%%) Gene:%d/%d", x, progress, gene_idx, #genome), "#44FF44")
            gui.text(2, 12, string.format("Action: %s", action_name), "#FFFFFF")
            gui.text(2, 22, string.format("Frame: %d", total_frames), "#888888")
            
            emu.frameadvance()
            total_frames = total_frames + 1
        end
        
        -- Read state after action
        local x = get_mario_x()
        local y = memory.readbyte(ADDR.MARIO_Y)
        local vel_x = signed_byte(memory.readbyte(ADDR.VEL_X))
        local vel_y = signed_byte(memory.readbyte(ADDR.VEL_Y))
        local float_state = memory.readbyte(ADDR.FLOAT_STATE)
        local on_ground = (float_state == 0) and 1 or 0
        local power = memory.readbyte(ADDR.POWER)
        local delta_x = x - prev_x
        local delta_y = y - prev_y
        local is_jumping = (action_id == 3 or action_id == 4 or action_id == 5 or 
                           action_id == 9 or action_id == 10) and 1 or 0
        local is_airborne = (float_state ~= 0) and 1 or 0
        
        -- Track max
        if x > max_x then max_x = x end
        
        -- Write CSV row
        csv:write(string.format("%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
            gene_idx, total_frames, action_id, action_name,
            x, y, vel_x, vel_y, float_state, on_ground, power,
            delta_x, delta_y, is_jumping, is_airborne))
        
        -- Track jumps (airborne transitions)
        if float_state ~= 0 and on_ground == 0 then
            if not jump_start_x then
                jump_start_x = prev_x
                jump_start_frame = gene_idx
                jump_peak_y = y
            end
            if y < jump_peak_y then jump_peak_y = y end
        else
            if jump_start_x then
                -- Jump just ended
                local jump_dist = x - jump_start_x
                local jump_height = prev_y - jump_peak_y  -- positive = went up
                table.insert(jumps, {
                    start_x = jump_start_x,
                    end_x = x,
                    distance = jump_dist,
                    height = jump_height,
                    start_gene = jump_start_frame,
                    end_gene = gene_idx,
                    peak_y = jump_peak_y,
                })
                jump_start_x = nil
                jump_start_frame = nil
                jump_peak_y = 999
            end
        end
        
        -- Track action distribution by X zone (50-pixel buckets)
        local zone = math.floor(x / 50) * 50
        if not action_at_x[zone] then action_at_x[zone] = {} end
        action_at_x[zone][action_id] = (action_at_x[zone][action_id] or 0) + 1
        
        prev_x = x
        prev_y = y
        
        -- Check termination
        if is_dead() then
            print(string.format("  DIED at x=%d, gene=%d", x, gene_idx))
            break
        end
        if x >= 3100 then
            print(string.format("  COMPLETED at x=%d, gene=%d, frames=%d", x, gene_idx, total_frames))
            break
        end
    end
    
    csv:close()
    print(string.format("CSV saved: genome_analysis.csv (%d rows)", total_frames / FRAME_SKIP))
    
    -- ========================
    -- GENERATE SUMMARY
    -- ========================
    local summary = io.open("genome_analysis_summary.txt", "w")
    
    summary:write("========================================\n")
    summary:write("WINNING GENOME ANALYSIS - SMB World 1-1\n")
    summary:write("========================================\n\n")
    summary:write(string.format("Total actions: %d\n", #genome))
    summary:write(string.format("Total frames: %d (%.1f seconds)\n", total_frames, total_frames / 60))
    summary:write(string.format("Max X reached: %d (%.1f%%)\n\n", max_x, max_x / 3168 * 100))
    
    -- Jump analysis
    summary:write("=== JUMP LOG ===\n")
    summary:write(string.format("Total jumps: %d\n\n", #jumps))
    summary:write(string.format("%-6s %-8s %-8s %-8s %-8s %-10s\n", 
        "Jump#", "Start X", "End X", "Dist", "Height", "Genes"))
    summary:write(string.rep("-", 55) .. "\n")
    
    for i, j in ipairs(jumps) do
        summary:write(string.format("%-6d %-8d %-8d %-8d %-8d %d-%d\n",
            i, j.start_x, j.end_x, j.distance, j.height,
            j.start_gene, j.end_gene))
    end
    
    -- Identify pit-crossing jumps (distance > 30 and height > 5)
    summary:write("\n=== PIT-CROSSING JUMPS (dist>30, height>5) ===\n")
    for i, j in ipairs(jumps) do
        if j.distance > 30 and j.height > 5 then
            summary:write(string.format(
                "  Jump %d: x=%d->%d (dist=%d, height=%d) at genes %d-%d\n",
                i, j.start_x, j.end_x, j.distance, j.height,
                j.start_gene, j.end_gene))
        end
    end
    
    -- Action distribution by zone
    summary:write("\n=== ACTION DISTRIBUTION BY ZONE ===\n")
    summary:write(string.format("%-8s ", "Zone"))
    for a = 0, 11 do
        summary:write(string.format("%-4s ", tostring(a)))
    end
    summary:write("Dominant\n")
    summary:write(string.rep("-", 70) .. "\n")
    
    local sorted_zones = {}
    for zone, _ in pairs(action_at_x) do table.insert(sorted_zones, zone) end
    table.sort(sorted_zones)
    
    for _, zone in ipairs(sorted_zones) do
        local acts = action_at_x[zone]
        local dominant_action = 0
        local dominant_count = 0
        summary:write(string.format("x%-7d ", zone))
        for a = 0, 11 do
            local c = acts[a] or 0
            summary:write(string.format("%-4d ", c))
            if c > dominant_count then
                dominant_count = c
                dominant_action = a
            end
        end
        summary:write(ACTION_NAMES[dominant_action] .. "\n")
    end
    
    summary:write("\n=== KEY INSIGHTS ===\n")
    summary:write("Look for:\n")
    summary:write("- Jump actions (3,4,5,9,10) near pit X positions = pit crossings\n")
    summary:write("- Zones with action changes = obstacle reactions\n")
    summary:write("- Long sequences of action 9 (Right+Jump+Run) = safe running\n")
    summary:write("- Zones where the agent slows down (Left actions) = obstacle approach\n")
    
    summary:close()
    print("Summary saved: genome_analysis_summary.txt")
    print("=== Analysis complete ===")
end

-- ========================
-- MAIN
-- ========================
local genome = load_genome()
if genome then
    analyze_genome(genome)
end
