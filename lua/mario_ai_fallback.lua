--[[
Super Mario Bros AI Training System - FCEUX Lua Script (Fallback Version)
========================================================================

This is a fallback version that uses file-based communication instead of WebSocket
when LuaSocket is not available in FCEUX.

Communication Protocol:
- Lua writes game state to: temp/game_state.json
- Lua reads actions from: temp/action.json
- Python trainer monitors these files for changes
]]

-- ============================================================================
-- CONFIGURATION AND CONSTANTS
-- ============================================================================

local CONFIG = {
    -- File-based communication settings
    GAME_STATE_FILE = "temp/game_state.json",
    ACTION_FILE = "temp/action.json",
    STATUS_FILE = "temp/status.json",
    
    -- Timing settings
    FRAME_DELAY_MS = 16,  -- ~60 FPS
    FILE_CHECK_INTERVAL = 1,  -- Check files every frame
    
    -- Debug settings
    DEBUG_ENABLED = true,
    LOG_TO_FILE = true,
    LOG_FILE_PATH = "logs/lua_fallback.log",
}

-- Action space mapping (12 actions total)
local ACTION_MAPPING = {
    [0] = {A=false, B=false, up=false, down=false, left=false, right=false, start=false, select=false}, -- No action
    [1] = {A=false, B=false, up=false, down=false, left=false, right=true, start=false, select=false},  -- Right
    [2] = {A=false, B=false, up=false, down=false, left=true, right=false, start=false, select=false},  -- Left
    [3] = {A=true, B=false, up=false, down=false, left=false, right=false, start=false, select=false},  -- Jump (A)
    [4] = {A=true, B=false, up=false, down=false, left=false, right=true, start=false, select=false},   -- Right + Jump
    [5] = {A=true, B=false, up=false, down=false, left=true, right=false, start=false, select=false},   -- Left + Jump
    [6] = {A=false, B=true, up=false, down=false, left=false, right=false, start=false, select=false},  -- Run/Fire (B)
    [7] = {A=false, B=true, up=false, down=false, left=false, right=true, start=false, select=false},   -- Right + Run
    [8] = {A=false, B=true, up=false, down=false, left=true, right=false, start=false, select=false},   -- Left + Run
    [9] = {A=true, B=true, up=false, down=false, left=false, right=true, start=false, select=false},    -- Right + Jump + Run
    [10] = {A=true, B=true, up=false, down=false, left=true, right=false, start=false, select=false},   -- Left + Jump + Run
    [11] = {A=false, B=false, up=false, down=true, left=false, right=false, start=false, select=false}  -- Crouch/Down
}

-- ============================================================================
-- GLOBAL STATE VARIABLES
-- ============================================================================

local g_state = {
    frame_id = 0,
    episode_id = 1,
    training_active = false,
    last_action_time = 0,
    pending_action = nil,
    log_file = nil,
}

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

-- Get current timestamp in milliseconds
local function get_timestamp_ms()
    return math.floor(os.clock() * 1000)
end

-- Initialize log file
local function init_log_file()
    if not CONFIG.LOG_TO_FILE then return end
    
    -- Create logs directory if it doesn't exist
    os.execute("mkdir logs 2>nul")  -- Windows command, ignore errors
    
    -- Open log file for writing
    g_state.log_file = io.open(CONFIG.LOG_FILE_PATH, "a")
    if g_state.log_file then
        g_state.log_file:write("\n" .. string.rep("=", 80) .. "\n")
        g_state.log_file:write("Super Mario Bros AI Training System - Lua Fallback Script\n")
        g_state.log_file:write("Session started: " .. os.date("%Y-%m-%d %H:%M:%S") .. "\n")
        g_state.log_file:write(string.rep("=", 80) .. "\n\n")
        g_state.log_file:flush()
    end
end

-- Debug logging function
local function debug_log(message, level)
    if not CONFIG.DEBUG_ENABLED then return end
    
    level = level or "INFO"
    local timestamp = get_timestamp_ms()
    local date_str = os.date("%H:%M:%S")
    local log_message = string.format("[%s] [%d] [%s] %s", date_str, timestamp, level, message)
    
    -- Console output
    print(log_message)
    
    -- FCEUX console output
    if emu and emu.print then
        emu.print(log_message)
    end
    
    -- File output
    if CONFIG.LOG_TO_FILE and g_state.log_file then
        g_state.log_file:write(log_message .. "\n")
        g_state.log_file:flush()
    end
end

-- Simple JSON encoder
local function encode_json(obj)
    if type(obj) == "table" then
        local parts = {}
        for k, v in pairs(obj) do
            local key = '"' .. tostring(k) .. '"'
            local value
            if type(v) == "string" then
                value = '"' .. tostring(v) .. '"'
            elseif type(v) == "number" or type(v) == "boolean" then
                value = tostring(v)
            elseif type(v) == "table" then
                value = encode_json(v)  -- Recursive encoding
            else
                value = '"' .. tostring(v) .. '"'
            end
            table.insert(parts, key .. ":" .. value)
        end
        return "{" .. table.concat(parts, ",") .. "}"
    else
        return '"' .. tostring(obj) .. '"'
    end
end

-- Simple JSON decoder
local function decode_json(str)
    local obj = {}
    
    -- Handle simple key-value pairs
    for key, value in string.gmatch(str, '"([^"]+)"%s*:%s*"?([^",}]+)"?') do
        if tonumber(value) then
            obj[key] = tonumber(value)
        elseif value == "true" then
            obj[key] = true
        elseif value == "false" then
            obj[key] = false
        else
            obj[key] = value
        end
    end
    
    return obj
end

-- ============================================================================
-- MEMORY READING FUNCTIONS
-- ============================================================================

-- Read Mario's core state information
local function read_mario_state()
    return {
        x_pos_screen = memory.readbyte(0x006D),
        x_pos_level = memory.readbyte(0x0086),
        x_pos_level_high = memory.readbyte(0x03AD),
        y_pos_screen = memory.readbyte(0x00CE),
        y_pos_level = memory.readbyte(0x03B8),
        x_velocity = memory.readbyte(0x0057),
        y_velocity = memory.readbyte(0x009F),
        direction = memory.readbyte(0x0045),
        on_ground = memory.readbyte(0x001D),
        power_state = memory.readbyte(0x0756),
        lives = memory.readbyte(0x075A),
        invincibility_timer = memory.readbyte(0x079E),
        animation_state = memory.readbyte(0x0079),
    }
end

-- Read level and world information
local function read_level_info()
    return {
        world_number = memory.readbyte(0x075F),
        level_number = memory.readbyte(0x0760),
        timer_hundreds = memory.readbyte(0x071A),
        timer_tens = memory.readbyte(0x071B),
        timer_ones = memory.readbyte(0x071C),
        screen_x_high = memory.readbyte(0x03AD),
        screen_x_low = memory.readbyte(0x0086),
    }
end

-- Read score and collectibles
local function read_score_info()
    return {
        score_100k = memory.readbyte(0x07DD),
        score_10k = memory.readbyte(0x07DE),
        score_1k = memory.readbyte(0x07DF),
        score_100 = memory.readbyte(0x07E0),
        score_10 = memory.readbyte(0x07E1),
        score_1 = memory.readbyte(0x07E2),
        coins_tens = memory.readbyte(0x075E),
        coins_ones = memory.readbyte(0x075D),
    }
end

-- Read game state
local function read_game_state()
    return {
        game_engine_state = memory.readbyte(0x0770),
        player_state = memory.readbyte(0x001D),
        game_mode = memory.readbyte(0x000E),
        end_of_level_flag = memory.readbyte(0x0772),
    }
end

-- Extract complete game state
local function extract_complete_game_state()
    local mario = read_mario_state()
    local level = read_level_info()
    local score = read_score_info()
    local game = read_game_state()
    
    -- Calculate derived values
    local mario_x_world = mario.x_pos_level + (mario.x_pos_level_high * 256)
    local total_score = (score.score_100k * 100000) + (score.score_10k * 10000) + 
                       (score.score_1k * 1000) + (score.score_100 * 100) + 
                       (score.score_10 * 10) + score.score_1
    local total_coins = (score.coins_tens * 10) + score.coins_ones
    local time_remaining = (level.timer_hundreds * 100) + (level.timer_tens * 10) + level.timer_ones
    
    -- Check for terminal conditions
    local is_dead = mario.lives == 0 or game.player_state == 0x0B
    local is_level_complete = game.end_of_level_flag == 1
    local is_time_up = time_remaining == 0
    
    return {
        frame_id = g_state.frame_id,
        timestamp = get_timestamp_ms(),
        episode_id = g_state.episode_id,
        mario_x = mario_x_world,
        mario_y = mario.y_pos_level,
        mario_velocity_x = mario.x_velocity,
        mario_velocity_y = mario.y_velocity,
        mario_power_state = mario.power_state,
        mario_lives = mario.lives,
        mario_on_ground = mario.on_ground,
        score = total_score,
        coins = total_coins,
        time = time_remaining,
        world = level.world_number,
        level = level.level_number,
        is_dead = is_dead,
        is_level_complete = is_level_complete,
        is_time_up = is_time_up,
        is_terminal = is_dead or is_level_complete or is_time_up,
    }
end

-- ============================================================================
-- FILE-BASED COMMUNICATION
-- ============================================================================

-- Create temp directory
local function create_temp_dir()
    os.execute("mkdir temp 2>nul")  -- Windows command, ignore errors
end

-- Write game state to file
local function write_game_state(game_state)
    local file = io.open(CONFIG.GAME_STATE_FILE, "w")
    if file then
        file:write(encode_json(game_state))
        file:close()
        return true
    else
        debug_log("Failed to write game state file", "ERROR")
        return false
    end
end

-- Read action from file
local function read_action()
    local file = io.open(CONFIG.ACTION_FILE, "r")
    if file then
        local content = file:read("*all")
        file:close()
        
        if content and content ~= "" then
            local action_data = decode_json(content)
            return action_data
        end
    end
    return nil
end

-- Write status to file
local function write_status(status)
    local file = io.open(CONFIG.STATUS_FILE, "w")
    if file then
        file:write(encode_json(status))
        file:close()
        return true
    else
        debug_log("Failed to write status file", "ERROR")
        return false
    end
end

-- Execute controller input
local function execute_controller_input(buttons)
    joypad.set(1, buttons)
    
    local button_str = ""
    for button, pressed in pairs(buttons) do
        if pressed then
            button_str = button_str .. button .. " "
        end
    end
    debug_log("Controller input: " .. (button_str ~= "" and button_str or "none"))
end

-- Convert action ID to controller input
local function action_to_controller_input(action_id)
    return ACTION_MAPPING[action_id] or ACTION_MAPPING[0]
end

-- ============================================================================
-- MAIN PROCESSING FUNCTIONS
-- ============================================================================

-- Process frame
local function process_frame()
    if not g_state.training_active then
        return
    end
    
    -- Extract game state
    local game_state = extract_complete_game_state()
    
    -- Write game state to file
    if not write_game_state(game_state) then
        debug_log("Failed to write game state", "ERROR")
        return
    end
    
    -- Read action from file
    local action_data = read_action()
    if action_data then
        local buttons
        
        if action_data.action ~= nil then
            -- Convert action ID to buttons
            buttons = action_to_controller_input(action_data.action)
            debug_log("Received action ID: " .. tostring(action_data.action))
        elseif action_data.buttons then
            -- Use buttons directly
            buttons = action_data.buttons
            debug_log("Received button states")
        else
            -- No action, use default
            buttons = ACTION_MAPPING[0]
            debug_log("No action received, using default")
        end
        
        -- Execute controller input
        execute_controller_input(buttons)
        
        -- Clear action file after processing
        local file = io.open(CONFIG.ACTION_FILE, "w")
        if file then
            file:write("")
            file:close()
        end
    else
        -- No action available, use no input
        execute_controller_input(ACTION_MAPPING[0])
    end
    
    -- Update frame counter
    g_state.frame_id = g_state.frame_id + 1
    
    -- Check for episode termination
    if game_state.is_terminal then
        debug_log("Episode terminated: " .. (game_state.is_dead and "death" or 
                 game_state.is_level_complete and "level_complete" or "time_up"))
        g_state.episode_id = g_state.episode_id + 1
        g_state.frame_id = 0
    end
    
    -- Log periodic status
    if g_state.frame_id % 60 == 0 then  -- Every second at 60 FPS
        debug_log(string.format("Episode %d, Frame %d: Mario X=%d, Score=%d, Lives=%d",
                  g_state.episode_id, g_state.frame_id, game_state.mario_x, 
                  game_state.score, game_state.mario_lives))
    end
end

-- Initialize system
local function initialize_system()
    debug_log("Initializing fallback communication system...")
    
    -- Initialize logging
    init_log_file()
    
    -- Create temp directory
    create_temp_dir()
    
    -- Write initial status
    local status = {
        connected = true,
        training_active = false,
        frame_id = 0,
        episode_id = 1,
        timestamp = get_timestamp_ms(),
        communication_method = "file_based"
    }
    write_status(status)
    
    -- Start training
    g_state.training_active = true
    
    debug_log("Fallback system initialized successfully")
    debug_log("Python trainer should monitor files in temp/ directory")
end

-- Cleanup system
local function cleanup_system()
    debug_log("Cleaning up fallback system...")
    
    g_state.training_active = false
    
    -- Write final status
    local status = {
        connected = false,
        training_active = false,
        timestamp = get_timestamp_ms(),
        reason = "script_shutdown"
    }
    write_status(status)
    
    -- Close log file
    if g_state.log_file then
        g_state.log_file:write("\nSession ended: " .. os.date("%Y-%m-%d %H:%M:%S") .. "\n")
        g_state.log_file:close()
        g_state.log_file = nil
    end
    
    debug_log("Cleanup completed")
end

-- ============================================================================
-- FCEUX INTEGRATION
-- ============================================================================

-- Main loop function called every frame by FCEUX
local function main_loop()
    process_frame()
end

-- FCEUX event handlers
local function on_frame_start()
    main_loop()
end

-- Register FCEUX callbacks
if emu then
    debug_log("Registering FCEUX callbacks...")
    
    -- Initialize system
    initialize_system()
    
    -- Register frame callback
    if emu.registerafter then
        emu.registerafter(on_frame_start)
    elseif emu.registerframe then
        emu.registerframe(on_frame_start)
    end
    
    -- Register exit callback
    if emu.registerexit then
        emu.registerexit(cleanup_system)
    end
    
    debug_log("FCEUX callbacks registered successfully")
else
    debug_log("FCEUX emu object not available, running in test mode")
    initialize_system()
end

-- ============================================================================
-- MANUAL CONTROL FUNCTIONS
-- ============================================================================

-- Manual functions for console control
function start_fallback_training()
    initialize_system()
    return true
end

function stop_fallback_training()
    cleanup_system()
    return true
end

function get_fallback_status()
    return {
        training_active = g_state.training_active,
        frame_id = g_state.frame_id,
        episode_id = g_state.episode_id,
    }
end

-- ============================================================================
-- SCRIPT ENTRY POINT
-- ============================================================================

debug_log("Super Mario Bros AI Training System - Fallback Version Loaded")
debug_log("This version uses file-based communication instead of WebSocket")
debug_log("Files are written to temp/ directory")
debug_log("Use start_fallback_training() to begin")

-- End of script
debug_log("mario_ai_fallback.lua script loaded successfully")