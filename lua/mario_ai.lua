
--[[
Super Mario Bros AI Training System - FCEUX Lua Script
=====================================================

This script implements the emulator-side component of a 2-part AI system for training
neural networks to play Super Mario Bros. It handles:
- WebSocket communication with Python trainer
- Memory reading from all specified Super Mario Bros addresses
- Control input execution (12-action space)
- Frame synchronization and timing control
- Error handling and reconnection logic

Author: AI Training System
Version: 1.0
Protocol Version: 1.0
]]

-- ============================================================================
-- CONFIGURATION AND CONSTANTS
-- ============================================================================

local CONFIG = {
    -- WebSocket connection settings
    WEBSOCKET_HOST = "localhost",
    WEBSOCKET_PORT = 8765,
    PROTOCOL_VERSION = "1.0",
    
    -- Frame synchronization settings
    FRAME_TIMEOUT_MS = 10000,  -- Increased to 1000ms for better stability
    MAX_FRAME_SKIP = 10,      -- Increased to 10 for more tolerance
    HEARTBEAT_INTERVAL = 5000, -- Increased to 5 seconds
    
    -- Reconnection settings
    MAX_RECONNECT_ATTEMPTS = 5,  -- Increased attempts
    RECONNECT_DELAY_MS = 2000,   -- Increased delay
    
    -- Connection health settings
    CONNECTION_HEALTH_CHECK_INTERVAL = 10000, -- Check every 10 seconds
    MAX_CONSECUTIVE_TIMEOUTS = 50,  -- Allow more timeouts before reconnection
    
    -- Debug settings
    DEBUG_ENABLED = false,
    LOG_MEMORY_READS = false,
    LOG_FRAME_SYNC = false,  -- Reduced logging for performance
    
    -- Logging settings
    LOG_TO_FILE = true,
    LOG_FILE_PATH = "logs/lua_debug.log",
    MAX_LOG_SIZE = 10 * 1024 * 1024,  -- 10MB max log file size
    
    -- Enhanced memory reading features
    ENHANCED_MEMORY_ENABLED = true,     -- Enable enhanced memory reading
    ENEMY_DETECTION_ENABLED = true,     -- Enable enemy detection and tracking
    POWERUP_DETECTION_ENABLED = true,   -- Enable power-up detection
    TILE_SAMPLING_ENABLED = true,       -- Enable level tile sampling around Mario
    ENHANCED_DEATH_DETECTION = true,    -- Enable enhanced death detection
    VELOCITY_TRACKING_ENABLED = true    -- Enable Mario velocity tracking
}

-- ============================================================================
-- ENHANCED MEMORY ADDRESSES (from reference script)
-- ============================================================================

-- Comprehensive memory address mapping for Super Mario Bros
local MEMORY_ADDRESSES = {
    -- Mario Position and State (Enhanced)
    MARIO_SCREEN_X = 0x03AD,        -- Mario's screen X position (0-255)
    MARIO_SCREEN_Y = 0x03B8,        -- Mario's screen Y position
    MARIO_X_PAGE = 0x006D,          -- Mario's X page (high byte for world position)
    MARIO_X_SUB = 0x0086,           -- Mario's X sub-position (low byte for world position)
    MARIO_STATE = 0x000E,           -- Mario's player state (0x0B=dying, 0x06=dead)
    MARIO_BELOW_VIEWPORT = 0x00B5,  -- Mario below viewport (>1 = in pit)
    MARIO_POWER = 0x0754,           -- Mario's power state (0=small, 1=big, 2=fire)
    MARIO_FACING = 0x0045,          -- Mario's facing direction (0=left, 1=right)
    MARIO_VELOCITY_X = 0x007B,      -- Mario's X velocity
    MARIO_VELOCITY_Y = 0x007D,      -- Mario's Y velocity
    
    -- Game State (Enhanced)
    LEVEL_TIME = 0x07F8,            -- Level timer (2 bytes)
    LEVEL_TIME_HIGH = 0x07F9,       -- Level timer high byte
    LIVES = 0x075A,                 -- Number of lives
    COINS = 0x075E,                 -- Number of coins
    SCORE_1 = 0x0758,               -- Current score byte 1 (BCD format)
    SCORE_100 = 0x0759,             -- Current score byte 2 (BCD format)
    SCORE_10K = 0x075A,             -- Current score byte 3 (BCD format)
    WORLD = 0x075F,                 -- Current world
    LEVEL = 0x0760,                 -- Current level
    GAME_STATE = 0x0770,            -- Game started flag
    END_OF_LEVEL_FLAG = 0x0772,     -- End of level flag
    IS_2PLAYER = 0x077A,            -- 2-player mode flag
    ONEUP_FLAG = 0x0772,            -- 1-up flag
    
    -- Enemy Detection (5 slots) - Enhanced from reference script
    ENEMY_SLOTS = {0x0F, 0x10, 0x11, 0x12, 0x13},     -- Enemy type in each slot
    ENEMY_X_PAGE = {0x6E, 0x6F, 0x70, 0x71, 0x72},    -- Enemy X pages
    ENEMY_X_SUB = {0x87, 0x88, 0x89, 0x8A, 0x8B},     -- Enemy X sub-positions
    ENEMY_Y = {0xCF, 0xD0, 0xD1, 0xD2, 0xD3},         -- Enemy Y positions
    ENEMY_STATE = {0x1E, 0x1F, 0x20, 0x21, 0x22},     -- Enemy states
    ENEMY_VELOCITY_X = {0x57, 0x58, 0x59, 0x5A, 0x5B}, -- Enemy X velocities
    ENEMY_VELOCITY_Y = {0x6D, 0x6E, 0x6F, 0x70, 0x71}, -- Enemy Y velocities
    ENEMY_DIRECTION = {0x45, 0x46, 0x47, 0x48, 0x49},  -- Enemy facing directions
    
    -- Power-up Detection (Enhanced)
    POWERUP_TYPE = 0x0039,          -- Power-up type
    POWERUP_X_PAGE = 0x008F,        -- Power-up X page
    POWERUP_X_SUB = 0x008F,         -- Power-up X sub-position
    POWERUP_Y = 0x00D7,             -- Power-up Y position
    POWERUP_STATE = 0x001E,         -- Power-up state
    
    -- Level Tile Data (Enhanced)
    LEVEL_LAYOUT = 0x0500,          -- Level tile data starts here
    TILE_DATA_SIZE = 0x0D * 0x10,   -- 13 rows x 16 columns per page
    
    -- Controller Input
    CONTROLLER_INPUT = 0x00F7,      -- Current controller input
    CONTROLLER_PREVIOUS = 0x00F6,   -- Previous controller input
    
    -- Camera/Screen Position
    SCREEN_X_HIGH = 0x03AD,         -- Camera page
    SCREEN_X_LOW = 0x0086,          -- Camera pixel position
    SCREEN_Y = 0x00B5,              -- Screen Y position
    VERTICAL_SCROLL = 0x0725        -- Vertical scroll position
}

-- Enemy type constants for enhanced detection
local ENEMY_TYPES = {
    NONE = 0x00,
    GOOMBA = 0x01,
    KOOPA = 0x02,
    BUZZY_BEETLE = 0x03,
    HAMMER_BRO = 0x04,
    LAKITU = 0x05,
    SPINY = 0x06,
    PIRANHA_PLANT = 0x07,
    BLOOPER = 0x08,
    BULLET_BILL = 0x09,
    CHEEP_CHEEP = 0x0A
}

-- Power-up type constants
local POWERUP_TYPES = {
    NONE = 0x00,
    MUSHROOM = 0x01,
    FIRE_FLOWER = 0x02,
    STAR = 0x03,
    ONE_UP = 0x04
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
    -- Connection state
    websocket = nil,
    connected = false,
    reconnect_attempts = 0,
    last_heartbeat = 0,
    last_reconnect_attempt = 0,
    reconnect_delay = 0,
    
    -- Connection health tracking
    last_health_check = 0,
    consecutive_timeouts = 0,
    successful_messages = 0,
    last_successful_message = 0,
    
    -- Frame synchronization
    frame_id = 0,
    waiting_for_action = false,
    last_frame_time = 0,
    frame_timeout_start = 0,
    pending_action = nil,
    
    -- Game state
    episode_id = 0,
    training_active = false,
    last_mario_x = 0,
    last_score = 0,
    
    -- Episode management
    episode_start_time = 0,
    episode_frames = 0,
    
    -- Error tracking
    desync_count = 0,
    error_count = 0,
    last_error_time = 0,
    
    -- Logging
    log_file = nil,
    log_file_size = 0
}

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

-- Lua 5.1 compatible bitwise operations
local function bit_and(a, b)
    local result = 0
    local bit_val = 1
    while a > 0 and b > 0 do
        if a % 2 == 1 and b % 2 == 1 then
            result = result + bit_val
        end
        a = math.floor(a / 2)
        b = math.floor(b / 2)
        bit_val = bit_val * 2
    end
    return result
end

local function bit_or(a, b)
    local result = 0
    local bit_val = 1
    while a > 0 or b > 0 do
        if (a % 2 == 1) or (b % 2 == 1) then
            result = result + bit_val
        end
        a = math.floor(a / 2)
        b = math.floor(b / 2)
        bit_val = bit_val * 2
    end
    return result
end

local function bit_xor(a, b)
    local result = 0
    local bit_val = 1
    while a > 0 or b > 0 do
        if (a % 2) ~= (b % 2) then
            result = result + bit_val
        end
        a = math.floor(a / 2)
        b = math.floor(b / 2)
        bit_val = bit_val * 2
    end
    return result
end

-- Get current wall-clock timestamp in milliseconds (not CPU time)
local function now_ms()
    if socket and socket.gettime then
        return math.floor(socket.gettime() * 1000)
    elseif emu and emu.getrealtimestate then
        local rt = emu.getrealtimestate()
        return rt and (rt.secs*1000 + math.floor(rt.usecs/1000)) or (os.time()*1000)
    else
        return os.time()*1000
    end
end

-- Initialize log file
local function init_log_file()
    if not CONFIG.LOG_TO_FILE then return end
    
    -- Create logs directory if it doesn't exist
    local log_dir = "logs"
    os.execute("mkdir " .. log_dir .. " 2>nul")  -- Windows command, ignore errors
    
    -- Open log file for writing
    g_state.log_file = io.open(CONFIG.LOG_FILE_PATH, "a")
    if g_state.log_file then
        g_state.log_file:write("\n" .. string.rep("=", 80) .. "\n")
        g_state.log_file:write("Super Mario Bros AI Training System - Lua Script Log\n")
        g_state.log_file:write("Session started: " .. os.date("%Y-%m-%d %H:%M:%S") .. "\n")
        g_state.log_file:write(string.rep("=", 80) .. "\n\n")
        g_state.log_file:flush()
        
        -- Get current file size
        local file_info = io.open(CONFIG.LOG_FILE_PATH, "r")
        if file_info then
            file_info:seek("end")
            g_state.log_file_size = file_info:seek()
            file_info:close()
        end
    end
end

-- Close log file
local function close_log_file()
    if g_state.log_file then
        g_state.log_file:write("\nSession ended: " .. os.date("%Y-%m-%d %H:%M:%S") .. "\n")
        g_state.log_file:write(string.rep("=", 80) .. "\n")
        g_state.log_file:close()
        g_state.log_file = nil
    end
end

-- Rotate log file if it gets too large
local function rotate_log_file()
    if not g_state.log_file or g_state.log_file_size < CONFIG.MAX_LOG_SIZE then
        return
    end
    
    -- Close current log file
    close_log_file()
    
    -- Rename current log to backup
    local backup_path = CONFIG.LOG_FILE_PATH .. ".bak"
    os.rename(CONFIG.LOG_FILE_PATH, backup_path)
    
    -- Reinitialize log file
    g_state.log_file_size = 0
    init_log_file()
end

-- Enhanced debug logging function with file output
local function debug_log(message, level)
    if not CONFIG.DEBUG_ENABLED then return end
    
    level = level or "INFO"
    local timestamp = now_ms()
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
        
        -- Update file size and check for rotation
        g_state.log_file_size = g_state.log_file_size + #log_message + 1
        if g_state.log_file_size > CONFIG.MAX_LOG_SIZE then
            rotate_log_file()
        end
    end
end

-- Convert signed byte value (handles two's complement)
local function signed_byte(value)
    if value > 127 then
        return value - 256
    end
    return value
end

-- Convert BCD (Binary Coded Decimal) to integer
local function bcd_to_int(bcd_value)
    local tens = math.floor(bcd_value / 16)
    local ones = bcd_value % 16
    return tens * 10 + ones
end

-- Calculate 16-bit value from two 8-bit values (little-endian)
local function read_u16_le(low_addr, high_addr)
    local low_byte = memory.readbyte(low_addr)
    local high_byte = memory.readbyte(high_addr)
    return low_byte + (high_byte * 256)
end

-- ============================================================================
-- MEMORY READING FUNCTIONS
-- ============================================================================

-- Global position tracking variables
local g_mario_true_x = 0  -- Track Mario's true X position across 255-pixel cycles
local g_previous_mario_x = 0  -- Previous frame's X position for cycle detection

-- Enhanced Mario state reading using comprehensive memory addresses
local function read_mario_state()
    -- Use enhanced memory addresses from MEMORY_ADDRESSES table
    local mario_screen_x = memory.readbyte(MEMORY_ADDRESSES.MARIO_SCREEN_X)      -- Mario's screen X position (0-255)
    local mario_screen_y = memory.readbyte(MEMORY_ADDRESSES.MARIO_SCREEN_Y)      -- Mario's screen Y position
    local mario_x_page = memory.readbyte(MEMORY_ADDRESSES.MARIO_X_PAGE)          -- Mario's X page (high byte for world position)
    local mario_x_sub = memory.readbyte(MEMORY_ADDRESSES.MARIO_X_SUB)            -- Mario's X sub-position (low byte for world position)
    local mario_player_state = memory.readbyte(MEMORY_ADDRESSES.MARIO_STATE)     -- Mario's player state (0x0B=dying, 0x06=dead)
    local mario_below_viewport = memory.readbyte(MEMORY_ADDRESSES.MARIO_BELOW_VIEWPORT) -- Mario below viewport (>1 = in pit)
    local mario_power_state = memory.readbyte(MEMORY_ADDRESSES.MARIO_POWER)      -- Mario's power state (0=small, 1=big, 2=fire)
    local mario_facing = memory.readbyte(MEMORY_ADDRESSES.MARIO_FACING)          -- Mario's facing direction
    local mario_velocity_x = memory.readbyte(MEMORY_ADDRESSES.MARIO_VELOCITY_X)  -- Mario's X velocity
    local mario_velocity_y = memory.readbyte(MEMORY_ADDRESSES.MARIO_VELOCITY_Y)  -- Mario's Y velocity
    local mario_lives = memory.readbyte(MEMORY_ADDRESSES.LIVES)                  -- Number of lives
    
    -- Calculate Mario's true world position using page and sub-position
    local mario_world_x = mario_x_page * 256 + mario_x_sub
    
    -- Update global tracking for consistency
    g_mario_true_x = mario_world_x
    g_previous_mario_x = mario_screen_x
    
    -- Debug: Log position tracking with improved accuracy
    if CONFIG.LOG_MEMORY_READS then
        debug_log(string.format("Mario position: screen_x=%d, world_x=%d, y=%d, page=%d, sub=%d",
                  mario_screen_x, mario_world_x, mario_screen_y, mario_x_page, mario_x_sub), "DEBUG")
    end
    
    -- Enhanced death detection
    local is_dying = (mario_player_state == 0x0B)
    local is_dead = (mario_player_state == 0x06)
    local is_in_pit = (mario_below_viewport > 1)
    
    local mario_data = {
        -- Position and movement using accurate addresses
        x_pos_raw = mario_screen_x,                  -- Screen X position (0-255)
        x_pos_world = mario_world_x,                 -- True world X position (page * 256 + sub)
        y_pos_level = mario_screen_y,                -- Screen Y position
        x_velocity = signed_byte(mario_velocity_x),  -- X velocity (signed)
        y_velocity = signed_byte(mario_velocity_y),  -- Y velocity (signed)
        direction = mario_facing,                    -- Facing direction (0=left, 1=right)
        
        -- Power state and status using accurate addresses
        power_state = mario_power_state,             -- Power state (0=small, 1=big, 2=fire)
        lives = mario_lives,                         -- Lives remaining
        player_state = mario_player_state,           -- Player state (0x0B=dying, 0x06=dead)
        invincibility_timer = 0,                     -- TODO: Find correct address
        animation_state = 0,                         -- TODO: Find correct address
        crouching = 0,                               -- TODO: Find correct address
        
        -- Enhanced state flags
        is_dying = is_dying,                         -- Mario is in dying animation
        is_dead = is_dead,                           -- Mario is dead
        is_in_pit = is_in_pit,                       -- Mario fell in pit
        below_viewport = mario_below_viewport,       -- Below viewport value
        
        -- Additional data
        x_page = mario_x_page,                       -- X page for reference
        x_sub = mario_x_sub,                         -- X sub-position for reference
    }
    
    -- Enhanced validation with better error detection
    if mario_data.y_pos_level > 240 then
        debug_log("Mario Y position seems invalid: " .. mario_data.y_pos_level, "WARN")
    end
    
    if mario_world_x < 0 or mario_world_x > 65535 then
        debug_log("Mario world X position seems invalid: " .. mario_world_x, "WARN")
    end
    
    if is_in_pit then
        debug_log("Mario is in pit! below_viewport=" .. mario_below_viewport, "DEBUG")
    end
    
    if is_dying or is_dead then
        debug_log(string.format("Mario death state: dying=%s, dead=%s, player_state=0x%02X",
                  tostring(is_dying), tostring(is_dead), mario_player_state), "DEBUG")
    end
    
    return mario_data
end

-- Level lengths for accurate progress calculation
local LEVEL_LEN = {
    ["1-1"]=3168, ["1-2"]=2816, ["1-3"]=3072, ["1-4"]=2048,
    -- Add more as needed
}

local function current_level_len(world, level)
    local key = tostring(world).."-"..tostring((level or 0)+1)
    return LEVEL_LEN[key] or 3168
end

-- Enhanced level and world information reading using enhanced addresses
local function read_level_info()
    -- Use enhanced timer reading (timer is stored as 2-byte word)
    local timer_word = memory.readbyte(MEMORY_ADDRESSES.LEVEL_TIME) + (memory.readbyte(MEMORY_ADDRESSES.LEVEL_TIME_HIGH) * 256)
    
    return {
        world_number = memory.readbyte(MEMORY_ADDRESSES.WORLD),          -- Current world
        level_number = memory.readbyte(MEMORY_ADDRESSES.LEVEL),          -- Current level
        
        -- Timer information
        time_remaining = timer_word,                                     -- Time remaining as word
        timer_hundreds = math.floor(timer_word / 100),
        timer_tens = math.floor((timer_word % 100) / 10),
        timer_ones = timer_word % 10,
        
        -- Screen/camera position (keep existing for compatibility)
        screen_x_high = memory.readbyte(MEMORY_ADDRESSES.SCREEN_X_HIGH), -- Camera page
        screen_x_low = memory.readbyte(MEMORY_ADDRESSES.SCREEN_X_LOW),   -- Mario's pixel position
        screen_y = memory.readbyte(MEMORY_ADDRESSES.SCREEN_Y),
        vertical_scroll = memory.readbyte(MEMORY_ADDRESSES.VERTICAL_SCROLL)
    }
end

-- Enhanced score reading with proper BCD handling using enhanced addresses
local function read_score_info()
    -- Score is stored in 3 bytes in BCD format
    local score1 = memory.readbyte(MEMORY_ADDRESSES.SCORE_1)
    local score2 = memory.readbyte(MEMORY_ADDRESSES.SCORE_100)
    local score3 = memory.readbyte(MEMORY_ADDRESSES.SCORE_10K)
    
    return {
        -- Individual score bytes (BCD format)
        score_1 = score1,
        score_100 = score2,
        score_10k = score3,
        score_100k = 0,  -- Placeholder for higher scores
        score_10 = 0,    -- Placeholder
        score_1k = 0,    -- Placeholder
        
        -- Coins
        coins_ones = memory.readbyte(MEMORY_ADDRESSES.COINS) % 10,
        coins_tens = math.floor(memory.readbyte(MEMORY_ADDRESSES.COINS) / 10),
        
        -- Additional flags
        oneup_flag = memory.readbyte(MEMORY_ADDRESSES.ONEUP_FLAG)
    }
end

-- Enhanced enemy reading with comprehensive detection using enhanced addresses
local function read_enemy_info()
    local enemies = {}
    
    -- Use enhanced memory addresses for enemy detection
    for i = 1, 5 do
        local enemy_type = memory.readbyte(MEMORY_ADDRESSES.ENEMY_SLOTS[i])
        local enemy_x_page = memory.readbyte(MEMORY_ADDRESSES.ENEMY_X_PAGE[i])
        local enemy_x_sub = memory.readbyte(MEMORY_ADDRESSES.ENEMY_X_SUB[i])
        local enemy_y = memory.readbyte(MEMORY_ADDRESSES.ENEMY_Y[i])
        local enemy_state = memory.readbyte(MEMORY_ADDRESSES.ENEMY_STATE[i])
        
        -- Enhanced: Read velocity and direction if enabled
        local enemy_x_velocity = 0
        local enemy_y_velocity = 0
        local enemy_direction = 0
        
        if CONFIG.ENHANCED_MEMORY_ENABLED and CONFIG.ENEMY_DETECTION_ENABLED then
            enemy_x_velocity = signed_byte(memory.readbyte(MEMORY_ADDRESSES.ENEMY_VELOCITY_X[i]))
            enemy_y_velocity = signed_byte(memory.readbyte(MEMORY_ADDRESSES.ENEMY_VELOCITY_Y[i]))
            enemy_direction = memory.readbyte(MEMORY_ADDRESSES.ENEMY_DIRECTION[i])
        end
        
        enemies[i-1] = {  -- 0-based indexing for compatibility
            type = enemy_type,
            x_pos = enemy_x_sub,  -- Screen relative position
            y_pos = enemy_y,
            x_world = enemy_x_page * 256 + enemy_x_sub,  -- World position
            state = enemy_state,
            x_velocity = enemy_x_velocity,  -- Enhanced: actual velocity
            y_velocity = enemy_y_velocity,  -- Enhanced: actual velocity
            direction = enemy_direction,    -- Enhanced: facing direction
            -- Additional enhanced data
            x_page = enemy_x_page,
            x_sub = enemy_x_sub,
            is_active = enemy_type > 0,
            threat_level = enemy_type > 0 and 1 or 0  -- Basic threat assessment
        }
    end
    
    -- Fill remaining slots for compatibility (8 total expected)
    for i = 5, 7 do
        enemies[i] = {
            type=0, x_pos=0, y_pos=0, x_world=0, state=0,
            x_velocity=0, y_velocity=0, direction=0,
            x_page=0, x_sub=0, is_active=false, threat_level=0
        }
    end
    
    return enemies
end

-- NEW: Enhanced power-up detection function
local function read_powerup_info()
    if not CONFIG.ENHANCED_MEMORY_ENABLED or not CONFIG.POWERUP_DETECTION_ENABLED then
        return {type=0, x_pos=0, y_pos=0, x_world=0, state=0, is_active=false}
    end
    
    local powerup_type = memory.readbyte(MEMORY_ADDRESSES.POWERUP_TYPE)
    local powerup_x_page = memory.readbyte(MEMORY_ADDRESSES.POWERUP_X_PAGE)
    local powerup_x_sub = memory.readbyte(MEMORY_ADDRESSES.POWERUP_X_SUB)
    local powerup_y = memory.readbyte(MEMORY_ADDRESSES.POWERUP_Y)
    local powerup_state = memory.readbyte(MEMORY_ADDRESSES.POWERUP_STATE)
    
    return {
        type = powerup_type,
        x_pos = powerup_x_sub,  -- Screen relative position
        y_pos = powerup_y,
        x_world = powerup_x_page * 256 + powerup_x_sub,  -- World position
        state = powerup_state,
        x_page = powerup_x_page,
        x_sub = powerup_x_sub,
        is_active = powerup_type > 0,
        powerup_name = powerup_type == POWERUP_TYPES.MUSHROOM and "Mushroom" or
                      powerup_type == POWERUP_TYPES.FIRE_FLOWER and "Fire Flower" or
                      powerup_type == POWERUP_TYPES.STAR and "Star" or
                      powerup_type == POWERUP_TYPES.ONE_UP and "1-Up" or "None"
    }
end

-- NEW: Level tile sampling around Mario for environmental awareness
local function read_level_tiles_around_mario(mario_world_x, mario_y)
    if not CONFIG.ENHANCED_MEMORY_ENABLED or not CONFIG.TILE_SAMPLING_ENABLED then
        return {}
    end
    
    local tiles = {}
    local sample_radius = 3  -- Sample 3 tiles in each direction
    
    -- Sample tiles in a grid around Mario
    for dy = -sample_radius, sample_radius do
        for dx = -sample_radius, sample_radius do
            local world_x = mario_world_x + dx * 16  -- Each tile is 16 pixels wide
            local world_y = mario_y + dy * 16        -- Each tile is 16 pixels tall
            
            -- Calculate tile address (simplified level layout)
            local page = math.floor(world_x / 256) % 2
            local subx = math.floor((world_x % 256) / 16)
            local suby = math.floor((world_y - 32) / 16)  -- Offset for ground level
            
            if suby >= 0 and suby < 13 and subx >= 0 and subx < 16 then
                local addr = MEMORY_ADDRESSES.LEVEL_LAYOUT + page * MEMORY_ADDRESSES.TILE_DATA_SIZE + suby * 16 + subx
                local tile_value = memory.readbyte(addr)
                
                local tile_key = string.format("%d_%d", dx, dy)
                tiles[tile_key] = {
                    x_offset = dx,
                    y_offset = dy,
                    tile_value = tile_value,
                    is_solid = tile_value > 0,
                    world_x = world_x,
                    world_y = world_y
                }
            end
        end
    end
    
    return tiles
end

-- NEW: Enhanced threat assessment for enemies
local function assess_enemy_threats(enemies, mario_world_x, mario_y)
    if not CONFIG.ENHANCED_MEMORY_ENABLED or not CONFIG.ENEMY_DETECTION_ENABLED then
        return {threat_count=0, nearest_threat_distance=999, threats_ahead=0, threats_behind=0}
    end
    
    local threat_count = 0
    local nearest_threat_distance = 999
    local threats_ahead = 0
    local threats_behind = 0
    
    for i = 0, 4 do  -- Check first 5 enemy slots
        local enemy = enemies[i]
        if enemy and enemy.is_active then
            local distance_x = math.abs(enemy.x_world - mario_world_x)
            local distance_y = math.abs(enemy.y_pos - mario_y)
            local total_distance = math.sqrt(distance_x * distance_x + distance_y * distance_y)
            
            -- Consider enemy a threat if within reasonable distance
            if total_distance < 200 then  -- 200 pixels
                threat_count = threat_count + 1
                
                if total_distance < nearest_threat_distance then
                    nearest_threat_distance = total_distance
                end
                
                -- Count threats ahead vs behind Mario
                if enemy.x_world > mario_world_x then
                    threats_ahead = threats_ahead + 1
                else
                    threats_behind = threats_behind + 1
                end
            end
        end
    end
    
    return {
        threat_count = threat_count,
        nearest_threat_distance = nearest_threat_distance,
        threats_ahead = threats_ahead,
        threats_behind = threats_behind
    }
end

-- Enhanced game state reading using enhanced addresses
local function read_game_state()
    return {
        game_engine_state = memory.readbyte(MEMORY_ADDRESSES.GAME_STATE),           -- Game started flag
        player_state = memory.readbyte(MEMORY_ADDRESSES.MARIO_STATE),               -- Mario's player state
        game_mode = memory.readbyte(MEMORY_ADDRESSES.GAME_STATE),                   -- Game mode (same as engine state)
        end_of_level_flag = memory.readbyte(MEMORY_ADDRESSES.END_OF_LEVEL_FLAG),    -- End of level flag
        controller_input = memory.readbyte(MEMORY_ADDRESSES.CONTROLLER_INPUT),      -- Current controller input
        controller_previous = memory.readbyte(MEMORY_ADDRESSES.CONTROLLER_PREVIOUS), -- Previous controller input
        is_2player = memory.readbyte(MEMORY_ADDRESSES.IS_2PLAYER)                   -- 2-player mode flag
    }
end

-- Read special objects (8 object slots)
local function read_objects_info()
    local objects = {}
    
    for i = 0, 7 do
        objects[i] = {
            type = memory.readbyte(0x0024 + i),
            x_pos = memory.readbyte(0x008F + i),
            y_pos = memory.readbyte(0x00D7 + i)
        }
    end
    
    return objects
end

-- Comprehensive game state extraction with enhanced features
local function extract_complete_game_state()
    local mario = read_mario_state()
    local level = read_level_info()
    local score = read_score_info()
    local enemies = read_enemy_info()
    local game = read_game_state()
    local objects = read_objects_info()
    
    -- NEW: Enhanced features
    local powerup = nil
    local level_tiles = {}
    local threat_assessment = {}
    
    if CONFIG.ENHANCED_MEMORY_ENABLED then
        if CONFIG.POWERUP_DETECTION_ENABLED then
            powerup = read_powerup_info()
        end
        
        if CONFIG.TILE_SAMPLING_ENABLED then
            level_tiles = read_level_tiles_around_mario(mario.x_pos_world or 0, mario.y_pos_level or 0)
        end
        
        if CONFIG.ENEMY_DETECTION_ENABLED then
            threat_assessment = assess_enemy_threats(enemies, mario.x_pos_world or 0, mario.y_pos_level or 0)
        end
    end
    
    -- Calculate derived values
    local total_score = (score.score_100k * 100000) + (score.score_10k * 10000) +
                       (score.score_1k * 1000) + (score.score_100 * 100) +
                       (score.score_10 * 10) + score.score_1
    
    local total_coins = (score.coins_tens * 10) + score.coins_ones
    
    -- Use BCD-decoded time from level info (don't recompute)
    local time_remaining = level.time_remaining
    
    -- Use consistent level length everywhere
    local level_len = current_level_len(level.world_number, level.level_number)
    local level_progress = math.max(0, math.min(1, (mario.x_pos_world or 0) / level_len))
    
    -- Enhanced episode end detection with improved death detection
    -- Initialize prev_lives on first run to prevent false death detection
    if g_state.prev_lives == nil then
        g_state.prev_lives = mario.lives
        debug_log(string.format("Initialized prev_lives to %d", mario.lives), "DEBUG")
    end
    
    local lost_life = mario.lives < g_state.prev_lives
    local is_dying = mario.is_dying or false  -- Mario is in dying animation
    local is_dead = mario.is_dead or false    -- Mario is dead
    local is_in_pit = mario.is_in_pit or false -- Mario fell in pit
    
    -- Enhanced death detection using multiple criteria with enhanced detection
    local death_detected = false
    if lost_life then
        death_detected = true
        debug_log("Death detected: Lost a life", "DEBUG")
    elseif is_dying then
        death_detected = true
        debug_log("Death detected: Mario is dying (player_state=0x0B)", "DEBUG")
    elseif is_dead then
        death_detected = true
        debug_log("Death detected: Mario is dead (player_state=0x06)", "DEBUG")
    elseif is_in_pit then
        death_detected = true
        debug_log("Death detected: Mario fell in pit (below_viewport>1)", "DEBUG")
    elseif mario.lives == 0 then
        death_detected = true
        debug_log("Death detected: No lives remaining", "DEBUG")
    end
    
    -- Enhanced death detection: Additional checks if enhanced features enabled
    if CONFIG.ENHANCED_MEMORY_ENABLED and CONFIG.ENHANCED_DEATH_DETECTION then
        -- Check for Mario being too far below screen
        if mario.y_pos_level > 250 then
            death_detected = true
            debug_log("Enhanced death detected: Mario Y position too low (" .. mario.y_pos_level .. ")", "DEBUG")
        end
        
        -- Check for Mario being stuck in a wall (velocity 0 for too long)
        -- This would require additional state tracking, implement if needed
    end
    
    -- Debug comprehensive death detection
    if death_detected then
        debug_log(string.format("Death details: lost_life=%s, dying=%s, dead=%s, in_pit=%s, lives=%d, player_state=0x%02X",
                  tostring(lost_life), tostring(is_dying), tostring(is_dead), tostring(is_in_pit),
                  mario.lives, mario.player_state), "DEBUG")
    end
    
    g_state.prev_lives = mario.lives
    
    -- Check for terminal conditions with improved detection
    local is_in_game = game.game_engine_state == 0x08 or game.game_engine_state == 0x00  -- In-game modes
    local is_level_complete = false
    local is_time_up = false
    
    if is_in_game then
        is_level_complete = game.end_of_level_flag == 1
        -- Don't use time as a terminal condition - let Mario play until death or level completion
        is_time_up = false  -- Disable time-based termination
    end
    
    return {
        -- Frame metadata
        frame_id = g_state.frame_id,
        timestamp = now_ms(),
        episode_id = g_state.episode_id,
        
        -- Mario state
        mario = mario,
        
        -- Level information
        level = level,
        level_length = level_len,  -- Add level length for reference
        
        -- Score and collectibles
        score = score,
        total_score = total_score,
        total_coins = total_coins,
        time_remaining = time_remaining,
        
        -- Enemies and objects
        enemies = enemies,
        objects = objects,
        
        -- Game state
        game = game,
        
        -- Derived values
        level_progress = level_progress,
        
        -- Terminal conditions
        is_dead = death_detected,
        is_level_complete = is_level_complete,
        is_time_up = is_time_up,
        is_terminal = death_detected or is_level_complete or is_time_up,
        
        -- NEW: Enhanced features (only included if enabled)
        powerup = powerup,
        level_tiles = level_tiles,
        threat_assessment = threat_assessment,
        
        -- Enhanced feature flags for debugging
        enhanced_features_enabled = CONFIG.ENHANCED_MEMORY_ENABLED,
        enemy_detection_enabled = CONFIG.ENEMY_DETECTION_ENABLED,
        powerup_detection_enabled = CONFIG.POWERUP_DETECTION_ENABLED,
        tile_sampling_enabled = CONFIG.TILE_SAMPLING_ENABLED,
        enhanced_death_detection = CONFIG.ENHANCED_DEATH_DETECTION,
        velocity_tracking_enabled = CONFIG.VELOCITY_TRACKING_ENABLED
    }
end

-- ============================================================================
-- BINARY DATA ENCODING (Lua 5.1 Compatible)
-- ============================================================================

-- Enhanced Lua 5.1 compatible pack functions with proper bounds checking
local function pack_u8(value)
    -- Ensure value is within valid range
    value = math.max(0, math.min(255, math.floor(value or 0)))
    return string.char(value)
end

local function pack_u16_le(value)
    -- Ensure value is within valid range for 16-bit unsigned
    value = math.max(0, math.min(65535, math.floor(value or 0)))
    local low = value % 256
    local high = math.floor(value / 256)
    return string.char(low, high)
end

local function pack_u16_be(value)
    -- Ensure value is within valid range for 16-bit unsigned
    value = math.max(0, math.min(65535, math.floor(value or 0)))
    local low = value % 256
    local high = math.floor(value / 256)
    return string.char(high, low)  -- Big-endian: high byte first
end

local function pack_u32_le(value)
    -- Ensure value is within valid range for 32-bit unsigned
    value = math.max(0, math.min(4294967295, math.floor(value or 0)))
    local b1 = value % 256
    local b2 = math.floor(value / 256) % 256
    local b3 = math.floor(value / 65536) % 256
    local b4 = math.floor(value / 16777216) % 256
    return string.char(b1, b2, b3, b4)
end

local function pack_i8(value)
    -- Handle signed 8-bit values (-128 to 127)
    value = math.max(-128, math.min(127, math.floor(value or 0)))
    if value < 0 then
        value = value + 256
    end
    return string.char(value)
end

-- Additional helper functions for consistent byte packing
local function pack_i16_le(value)
    -- Handle signed 16-bit values (-32768 to 32767)
    value = math.max(-32768, math.min(32767, math.floor(value or 0)))
    if value < 0 then
        value = value + 65536
    end
    local low = value % 256
    local high = math.floor(value / 256)
    return string.char(low, high)
end

local function pack_i32_le(value)
    -- Handle signed 32-bit values
    value = math.max(-2147483648, math.min(2147483647, math.floor(value or 0)))
    if value < 0 then
        value = value + 4294967296
    end
    local b1 = value % 256
    local b2 = math.floor(value / 256) % 256
    local b3 = math.floor(value / 65536) % 256
    local b4 = math.floor(value / 16777216) % 256
    return string.char(b1, b2, b3, b4)
end

-- Pack game state into binary format for efficient transmission
-- CRITICAL: This function MUST always produce exactly 128-byte payloads
local function pack_binary_game_state(game_state)
    local mario = game_state.mario
    local level = game_state.level
    local score = game_state.score
    
    -- Debug: Log Mario position values being packed
    debug_log(string.format("PACKING: raw_x=%d, true_x=%d, y=%d, lives=%d",
              mario.x_pos_raw or 0, mario.x_pos_world or 0, mario.y_pos_level or 0, mario.lives or 0), "DEBUG")
    
    -- Initialize payload buffer with exactly 128 zero bytes
    local payload_buffer = {}
    for i = 1, 128 do
        payload_buffer[i] = 0
    end
    
    local pos = 1  -- Current position in buffer (1-based indexing)
    
    -- Mario Data Block (16 bytes: positions 1-16)
    local mario_x_world = mario.x_pos_world or 0
    local mario_y_level = mario.y_pos_level or 0
    local mario_x_vel = mario.x_velocity or 0
    local mario_y_vel = mario.y_velocity or 0
    
    -- Pack Mario X position (2 bytes, little-endian)
    payload_buffer[pos] = mario_x_world % 256
    payload_buffer[pos + 1] = math.floor(mario_x_world / 256) % 256
    pos = pos + 2
    
    -- Pack Mario Y position (2 bytes, little-endian)
    payload_buffer[pos] = mario_y_level % 256
    payload_buffer[pos + 1] = math.floor(mario_y_level / 256) % 256
    pos = pos + 2
    
    -- Pack Mario X velocity (1 byte, signed)
    if mario_x_vel < 0 then
        payload_buffer[pos] = (mario_x_vel + 256) % 256
    else
        payload_buffer[pos] = mario_x_vel % 256
    end
    pos = pos + 1
    
    -- Pack Mario Y velocity (1 byte, signed)
    if mario_y_vel < 0 then
        payload_buffer[pos] = (mario_y_vel + 256) % 256
    else
        payload_buffer[pos] = mario_y_vel % 256
    end
    pos = pos + 1
    
    -- Pack Mario state data (10 bytes)
    payload_buffer[pos] = (mario.power_state or 0) % 256; pos = pos + 1
    payload_buffer[pos] = (mario.animation_state or 0) % 256; pos = pos + 1
    payload_buffer[pos] = (mario.direction or 1) % 256; pos = pos + 1
    payload_buffer[pos] = (mario.player_state or 0) % 256; pos = pos + 1
    payload_buffer[pos] = (mario.lives or 3) % 256; pos = pos + 1
    payload_buffer[pos] = (mario.invincibility_timer or 0) % 256; pos = pos + 1
    payload_buffer[pos] = (mario.x_pos_raw or 0) % 256; pos = pos + 1
    payload_buffer[pos] = (mario.crouching or 0) % 256; pos = pos + 1
    payload_buffer[pos] = 0; pos = pos + 1  -- Reserved
    payload_buffer[pos] = 0; pos = pos + 1  -- Reserved
    -- Mario block complete: 16 bytes (positions 1-16)
    
    -- Enemy Data Block (32 bytes: positions 17-48)
    for i = 0, 7 do
        local enemy = game_state.enemies[i] or {type=0, x_pos=0, y_pos=0, state=0}
        payload_buffer[pos] = (enemy.type or 0) % 256; pos = pos + 1
        payload_buffer[pos] = (enemy.x_pos or 0) % 256; pos = pos + 1
        payload_buffer[pos] = (enemy.y_pos or 0) % 256; pos = pos + 1
        payload_buffer[pos] = (enemy.state or 0) % 256; pos = pos + 1
    end
    -- Enemy block complete: 32 bytes (positions 17-48)
    
    -- Level Data Block (64 bytes: positions 49-112)
    local camera_x = (level.screen_x_low or 0) + ((level.screen_x_high or 0) * 256)
    
    -- Pack camera X position (2 bytes, little-endian)
    payload_buffer[pos] = camera_x % 256
    payload_buffer[pos + 1] = math.floor(camera_x / 256) % 256
    pos = pos + 2
    
    -- Pack level info (8 bytes)
    payload_buffer[pos] = (level.world_number or 1) % 256; pos = pos + 1
    payload_buffer[pos] = (level.level_number or 1) % 256; pos = pos + 1
    payload_buffer[pos] = (score.score_100k or 0) % 256; pos = pos + 1
    payload_buffer[pos] = (score.score_10k or 0) % 256; pos = pos + 1
    payload_buffer[pos] = (score.score_1k or 0) % 256; pos = pos + 1
    payload_buffer[pos] = (score.score_100 or 0) % 256; pos = pos + 1
    
    -- Pack time remaining (4 bytes, little-endian)
    local time_remaining = game_state.time_remaining or 0
    payload_buffer[pos] = time_remaining % 256
    payload_buffer[pos + 1] = math.floor(time_remaining / 256) % 256
    payload_buffer[pos + 2] = math.floor(time_remaining / 65536) % 256
    payload_buffer[pos + 3] = math.floor(time_remaining / 16777216) % 256
    pos = pos + 4
    
    -- Pack total coins (2 bytes, little-endian)
    local total_coins = game_state.total_coins or 0
    payload_buffer[pos] = total_coins % 256
    payload_buffer[pos + 1] = math.floor(total_coins / 256) % 256
    pos = pos + 2
    
    -- Enhanced features in remaining 48 bytes of level block (if enabled)
    if CONFIG.ENHANCED_MEMORY_ENABLED then
        -- Pack power-up information (8 bytes)
        local powerup = game_state.powerup or {type=0, x_pos=0, y_pos=0, x_world=0, state=0, is_active=false}
        payload_buffer[pos] = (powerup.type or 0) % 256; pos = pos + 1
        payload_buffer[pos] = (powerup.x_pos or 0) % 256; pos = pos + 1
        payload_buffer[pos] = (powerup.y_pos or 0) % 256; pos = pos + 1
        payload_buffer[pos] = (powerup.state or 0) % 256; pos = pos + 1
        -- Pack power-up world X position (2 bytes, little-endian)
        local powerup_world_x = powerup.x_world or 0
        payload_buffer[pos] = powerup_world_x % 256
        payload_buffer[pos + 1] = math.floor(powerup_world_x / 256) % 256
        pos = pos + 2
        payload_buffer[pos] = (powerup.is_active and 1 or 0) % 256; pos = pos + 1
        payload_buffer[pos] = 0; pos = pos + 1  -- Reserved for future power-up data
        
        -- Pack threat assessment (8 bytes)
        local threats = game_state.threat_assessment or {threat_count=0, nearest_threat_distance=999, threats_ahead=0, threats_behind=0}
        payload_buffer[pos] = (threats.threat_count or 0) % 256; pos = pos + 1
        payload_buffer[pos] = (threats.threats_ahead or 0) % 256; pos = pos + 1
        payload_buffer[pos] = (threats.threats_behind or 0) % 256; pos = pos + 1
        payload_buffer[pos] = 0; pos = pos + 1  -- Reserved
        -- Pack nearest threat distance (2 bytes, little-endian)
        local nearest_distance = math.min(65535, math.floor(threats.nearest_threat_distance or 999))
        payload_buffer[pos] = nearest_distance % 256
        payload_buffer[pos + 1] = math.floor(nearest_distance / 256) % 256
        pos = pos + 2
        payload_buffer[pos] = 0; pos = pos + 1  -- Reserved
        payload_buffer[pos] = 0; pos = pos + 1  -- Reserved
        
        -- Pack enhanced Mario velocity data (4 bytes)
        if CONFIG.VELOCITY_TRACKING_ENABLED then
            local mario_x_vel = mario.x_velocity or 0
            local mario_y_vel = mario.y_velocity or 0
            -- Pack signed velocities
            if mario_x_vel < 0 then
                payload_buffer[pos] = (mario_x_vel + 256) % 256
            else
                payload_buffer[pos] = mario_x_vel % 256
            end
            pos = pos + 1
            if mario_y_vel < 0 then
                payload_buffer[pos] = (mario_y_vel + 256) % 256
            else
                payload_buffer[pos] = mario_y_vel % 256
            end
            pos = pos + 1
            payload_buffer[pos] = (mario.direction or 1) % 256; pos = pos + 1  -- Facing direction
            payload_buffer[pos] = (mario.below_viewport or 0) % 256; pos = pos + 1  -- Enhanced death detection
        else
            pos = pos + 4  -- Skip velocity data if disabled
        end
        
        -- Pack level tile sampling data (16 bytes) - simplified representation
        if CONFIG.TILE_SAMPLING_ENABLED and game_state.level_tiles then
            -- Pack a 4x4 grid of tiles around Mario (16 bytes)
            local tile_offsets = {
                {-1, -1}, {0, -1}, {1, -1}, {2, -1},  -- Row above Mario
                {-1, 0},  {0, 0},  {1, 0},  {2, 0},   -- Mario's row
                {-1, 1},  {0, 1},  {1, 1},  {2, 1},   -- Row below Mario
                {-1, 2},  {0, 2},  {1, 2},  {2, 2}    -- Two rows below Mario
            }
            
            for i, offset in ipairs(tile_offsets) do
                local tile_key = string.format("%d_%d", offset[1], offset[2])
                local tile = game_state.level_tiles[tile_key]
                if tile then
                    payload_buffer[pos] = (tile.tile_value or 0) % 256
                else
                    payload_buffer[pos] = 0  -- No tile data
                end
                pos = pos + 1
            end
        else
            pos = pos + 16  -- Skip tile data if disabled
        end
        
        -- Pack enhanced enemy velocity data (12 bytes) - first 3 enemies
        if CONFIG.ENEMY_DETECTION_ENABLED then
            for i = 0, 2 do  -- First 3 enemies get enhanced data
                local enemy = game_state.enemies[i] or {x_velocity=0, y_velocity=0, direction=0, threat_level=0}
                -- Pack enemy velocities (signed)
                local enemy_x_vel = enemy.x_velocity or 0
                local enemy_y_vel = enemy.y_velocity or 0
                if enemy_x_vel < 0 then
                    payload_buffer[pos] = (enemy_x_vel + 256) % 256
                else
                    payload_buffer[pos] = enemy_x_vel % 256
                end
                pos = pos + 1
                if enemy_y_vel < 0 then
                    payload_buffer[pos] = (enemy_y_vel + 256) % 256
                else
                    payload_buffer[pos] = enemy_y_vel % 256
                end
                pos = pos + 1
                payload_buffer[pos] = (enemy.direction or 0) % 256; pos = pos + 1
                payload_buffer[pos] = (enemy.threat_level or 0) % 256; pos = pos + 1
            end
        else
            pos = pos + 12  -- Skip enhanced enemy data if disabled
        end
    else
        -- If enhanced features disabled, skip all 48 bytes
        pos = pos + 48
    end
    -- Level block complete: 64 bytes (positions 49-112)
    
    -- Game Variables Block (16 bytes: positions 113-128)
    payload_buffer[pos] = (game_state.game.game_engine_state or 0) % 256; pos = pos + 1
    payload_buffer[pos] = math.floor((game_state.level_progress or 0) * 100) % 256; pos = pos + 1
    
    -- Pack distance to flag (2 bytes, little-endian)
    local level_len = current_level_len(level.world_number or 1, level.level_number or 1)
    local distance_to_flag = math.max(0, level_len - (mario.x_pos_world or 0))
    payload_buffer[pos] = distance_to_flag % 256
    payload_buffer[pos + 1] = math.floor(distance_to_flag / 256) % 256
    pos = pos + 2
    
    -- Pack frame ID (4 bytes, little-endian)
    local frame_id = game_state.frame_id or 0
    payload_buffer[pos] = frame_id % 256
    payload_buffer[pos + 1] = math.floor(frame_id / 256) % 256
    payload_buffer[pos + 2] = math.floor(frame_id / 65536) % 256
    payload_buffer[pos + 3] = math.floor(frame_id / 16777216) % 256
    pos = pos + 4
    
    -- Pack timestamp (4 bytes, little-endian)
    local timestamp = now_ms()
    payload_buffer[pos] = timestamp % 256
    payload_buffer[pos + 1] = math.floor(timestamp / 256) % 256
    payload_buffer[pos + 2] = math.floor(timestamp / 65536) % 256
    payload_buffer[pos + 3] = math.floor(timestamp / 16777216) % 256
    pos = pos + 4
    
    -- Remaining 4 bytes in game block are reserved (already zeroed)
    -- Game block complete: 16 bytes (positions 113-128)
    
    -- Convert buffer to string (exactly 128 bytes)
    local payload_chars = {}
    for i = 1, 128 do
        payload_chars[i] = string.char(payload_buffer[i])
    end
    local payload = table.concat(payload_chars)
    
    -- Verify payload size is exactly 128 bytes
    if #payload ~= 128 then
        debug_log(string.format("CRITICAL ERROR: Payload size is %d bytes, expected exactly 128!", #payload), "ERROR")
        -- Force exactly 128 bytes
        if #payload < 128 then
            payload = payload .. string.rep("\0", 128 - #payload)
        else
            payload = string.sub(payload, 1, 128)
        end
    end
    
    debug_log(string.format("Payload size: %d bytes (EXACTLY 128 as required)", #payload), "DEBUG")
    
    -- Calculate simple checksum (sum of all payload bytes)
    local checksum = 0
    for i = 1, #payload do
        checksum = (checksum + string.byte(payload, i)) % 256
    end
    
    -- Create header (8 bytes) - data_length excludes header
    local header = ""
    header = header .. pack_u8(0x01)                   -- 1 byte: Message Type (0x01 = game_state)
    header = header .. pack_u32_le(game_state.frame_id) -- 4 bytes: Frame ID
    header = header .. pack_u16_le(128)                 -- 2 bytes: Data Length (ALWAYS 128)
    header = header .. pack_u8(checksum)               -- 1 byte: Checksum
    
    local packet = header .. payload
    debug_log(string.format("Total packet size: %d bytes (8-byte header + 128-byte payload = 136)", #packet), "DEBUG")
    
    return packet
end

-- ============================================================================
-- WEBSOCKET COMMUNICATION (SIMPLIFIED IMPLEMENTATION)
-- ============================================================================

-- Note: This is a simplified WebSocket implementation for FCEUX Lua
-- In a real implementation, you would need a proper WebSocket library

-- Try to load required libraries with detailed error handling
local socket_ok, socket_err = pcall(require, "socket")
local socket = nil

if socket_ok then
    socket = socket_err  -- pcall returns the module as second value on success
    debug_log("LuaSocket loaded successfully")
    
    -- Test basic socket functionality
    local test_socket = socket.tcp()
    if test_socket then
        debug_log("TCP socket creation successful")
        test_socket:close()
    else
        debug_log("TCP socket creation failed", "ERROR")
        socket = nil
    end
else
    debug_log("LuaSocket load failed: " .. tostring(socket_err), "ERROR")
    
    -- Check if socket.core is the issue
    local core_ok, core_err = pcall(require, "socket.core")
    if not core_ok then
        debug_log("socket.core not available: " .. tostring(core_err), "ERROR")
        debug_log("This usually means LuaSocket binary modules are missing", "ERROR")
    else
        debug_log("socket.core is available, trying direct core usage", "INFO")
        -- Try using socket.core directly
        socket = core_err
    end
    
    if not socket then
        debug_log("FCEUX does not have working LuaSocket support", "ERROR")
        debug_log("Cannot establish WebSocket connection", "ERROR")
    end
end

-- Simple JSON implementation (always use this for reliability)
local json = {
    encode = function(obj)
        if type(obj) == "table" then
            local parts = {}
            for k, v in pairs(obj) do
                local key = '"' .. tostring(k) .. '"'
                local value
                if type(v) == "string" then
                    value = '"' .. tostring(v) .. '"'
                elseif type(v) == "number" or type(v) == "boolean" then
                    value = tostring(v)
                else
                    value = '"' .. tostring(v) .. '"'
                end
                table.insert(parts, key .. ":" .. value)
            end
            return "{" .. table.concat(parts, ",") .. "}"
        else
            return '"' .. tostring(obj) .. '"'
        end
    end,
    decode = function(str)
        -- Enhanced JSON decoder to handle nested objects
        local obj = {}
        
        -- First, handle simple key-value pairs
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
        
        -- Handle nested objects (specifically for buttons field)
        local buttons_match = string.match(str, '"buttons"%s*:%s*({[^}]*})')
        if buttons_match then
            -- Parse the buttons object
            local buttons = {}
            for key, value in string.gmatch(buttons_match, '"([^"]+)"%s*:%s*([^,}]+)') do
                if value == "true" then
                    buttons[key] = true
                elseif value == "false" then
                    buttons[key] = false
                else
                    buttons[key] = value
                end
            end
            obj.buttons = buttons
        end
        
        return obj
    end
}

debug_log("Using built-in JSON implementation")

-- Initialize WebSocket connection
local function init_websocket()
    debug_log("Initializing WebSocket connection...")
    
    if not socket then
        debug_log("Cannot initialize WebSocket: LuaSocket not available", "ERROR")
        debug_log("Please ensure FCEUX has proper LuaSocket support", "ERROR")
        return nil
    end
    
    -- Test socket creation first
    local tcp_socket, socket_err = socket.tcp()
    if not tcp_socket then
        debug_log("Failed to create TCP socket: " .. (socket_err or "unknown error"), "ERROR")
        return nil
    end
    
    debug_log("TCP socket created successfully")
    
    -- Set connection timeout - longer for initial connection
    tcp_socket:settimeout(15) -- 15 second timeout for connection
    
    debug_log("Attempting to connect to " .. CONFIG.WEBSOCKET_HOST .. ":" .. CONFIG.WEBSOCKET_PORT)
    
    -- Try connection with better error handling
    local result, err = tcp_socket:connect(CONFIG.WEBSOCKET_HOST, CONFIG.WEBSOCKET_PORT)
    if not result then
        debug_log("Failed to connect to WebSocket server: " .. (err or "unknown error"), "ERROR")
        debug_log("Common causes:", "ERROR")
        debug_log("1. Python training system is not running", "ERROR")
        debug_log("2. Port 8765 is blocked or in use", "ERROR")
        debug_log("3. Firewall is blocking the connection", "ERROR")
        debug_log("4. Wrong host/port configuration", "ERROR")
        tcp_socket:close()
        return nil
    end
    
    debug_log("TCP connection established successfully!")
    debug_log("Performing WebSocket handshake...")
    
    -- Perform WebSocket handshake (simplified but more robust)
    local handshake = string.format(
        "GET / HTTP/1.1\r\n" ..
        "Host: %s:%d\r\n" ..
        "Upgrade: websocket\r\n" ..
        "Connection: Upgrade\r\n" ..
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n" ..
        "Sec-WebSocket-Version: 13\r\n" ..
        "Origin: http://%s:%d\r\n" ..
        "\r\n",
        CONFIG.WEBSOCKET_HOST, CONFIG.WEBSOCKET_PORT,
        CONFIG.WEBSOCKET_HOST, CONFIG.WEBSOCKET_PORT
    )
    
    debug_log("Sending WebSocket handshake...")
    local send_result, send_err = tcp_socket:send(handshake)
    if not send_result then
        debug_log("Failed to send WebSocket handshake: " .. (send_err or "unknown error"), "ERROR")
        tcp_socket:close()
        return nil
    end
    
    debug_log("Handshake sent, waiting for response...")
    
    -- Read handshake response with timeout
    tcp_socket:settimeout(10) -- 10 second timeout for response
    local response, recv_err = tcp_socket:receive("*l")
    if not response then
        debug_log("Failed to receive handshake response: " .. (recv_err or "unknown error"), "ERROR")
        debug_log("This usually means the server is not a WebSocket server", "ERROR")
        tcp_socket:close()
        return nil
    end
    
    debug_log("Received handshake response: " .. response)
    
    -- Check for successful WebSocket upgrade
    if not string.find(response, "101") or not string.find(response, "Switching Protocols") then
        debug_log("WebSocket handshake failed - invalid response", "ERROR")
        debug_log("Expected '101 Switching Protocols', got: " .. response, "ERROR")
        debug_log("The server may not support WebSocket connections", "ERROR")
        tcp_socket:close()
        return nil
    end
    
    debug_log("WebSocket handshake successful!")
    
    -- Skip remaining headers
    local header_count = 0
    repeat
        local line, line_err = tcp_socket:receive("*l")
        if line_err and line_err ~= "timeout" then
            debug_log("Error reading headers: " .. line_err, "WARN")
            break
        end
        header_count = header_count + 1
        if header_count > 20 then -- Prevent infinite loop
            debug_log("Too many headers, stopping", "WARN")
            break
        end
    until not line or line == ""
    
    -- Set non-blocking mode for game loop with faster timeout
    tcp_socket:settimeout(0.001) -- 1ms timeout for better performance
    
    debug_log("WebSocket connection established successfully!")
    debug_log("Connection is ready for communication")
    return tcp_socket
end

-- Send WebSocket message (simplified frame format with proper masking)
local function send_websocket_message(socket, data, is_binary)
    if not socket then return false end
    
    local opcode = is_binary and 0x02 or 0x01
    local payload_len = #data
    local frame = ""
    
    -- First byte: FIN (1) + RSV (000) + Opcode (4 bits)
    frame = frame .. pack_u8(bit_or(0x80, opcode))
    
    -- Second byte: MASK (1) + Payload length (7 bits)
    if payload_len < 126 then
        frame = frame .. pack_u8(bit_or(0x80, payload_len))  -- Set mask bit
    elseif payload_len < 65536 then
        frame = frame .. pack_u8(bit_or(0x80, 126))  -- Set mask bit + extended length
        frame = frame .. pack_u16_be(payload_len)  -- Big-endian for WebSocket
    else
        -- For very large payloads, truncate for simplicity
        frame = frame .. pack_u8(bit_or(0x80, 126))
        frame = frame .. pack_u16_be(65535)
        data = string.sub(data, 1, 65535)
        payload_len = 65535
    end
    
    -- Generate 4-byte masking key
    local mask_key = {
        math.random(0, 255),
        math.random(0, 255),
        math.random(0, 255),
        math.random(0, 255)
    }
    
    -- Add masking key to frame
    for i = 1, 4 do
        frame = frame .. pack_u8(mask_key[i])
    end
    
    -- Mask the payload
    local masked_data = ""
    for i = 1, payload_len do
        local data_byte = string.byte(data, i)
        local mask_byte = mask_key[((i - 1) % 4) + 1]
        masked_data = masked_data .. pack_u8(bit_xor(data_byte, mask_byte))
    end
    
    frame = frame .. masked_data
    
    local result, err = socket:send(frame)
    if not result then
        debug_log("Failed to send WebSocket message: " .. (err or "unknown error"), "ERROR")
        return false
    end
    
    return true
end

-- Receive EXACTLY N bytes from socket (handles partial reads in non-blocking mode)
local function recv_exact(sock, n)
    local buf, got = {}, 0
    local max_attempts = 100 -- Prevent infinite loops
    local attempts = 0
    
    while got < n and attempts < max_attempts do
        attempts = attempts + 1
        local chunk, err, partial = sock:receive(n - got)
        
        if chunk then
            buf[#buf+1] = chunk
            got = got + #chunk
        elseif partial and #partial > 0 then
            buf[#buf+1] = partial
            got = got + #partial
        elseif err == "timeout" then
            -- For timeout, just continue trying (non-blocking mode)
            if got == 0 then
                return nil, "timeout" -- No data received at all
            end
            -- Continue trying to get remaining bytes
        elseif err == "closed" then
            return nil, "closed"
        else
            return nil, err or "unknown error"
        end
    end
    
    if got < n then
        return nil, "incomplete"
    end
    
    return table.concat(buf)
end

-- Receive WebSocket message (robust for 60 FPS with better error handling)
local function receive_websocket_message(socket)
    if not socket then return nil end
    
    -- Set reasonable timeout for non-blocking operation
    socket:settimeout(0.01) -- 10ms timeout for better stability
    
    local header, err = recv_exact(socket, 2)
    if not header then
        if err == "timeout" then
            return nil -- Normal timeout, no error
        elseif err == "closed" then
            debug_log("WebSocket connection closed by server", "WARN")
            return nil, "closed"
        else
            debug_log("WebSocket receive error: " .. (err or "unknown error"), "ERROR")
            return nil, err
        end
    end
    
    local first_byte, second_byte = string.byte(header, 1, 2)
    local fin = bit_and(first_byte, 0x80) ~= 0
    local opcode = bit_and(first_byte, 0x0F)
    local masked = bit_and(second_byte, 0x80) ~= 0
    local payload_len = bit_and(second_byte, 0x7F)
    
    -- Handle close frame (opcode 0x08)
    if opcode == 0x08 then
        debug_log("Received WebSocket close frame - connection terminated by server", "WARN")
        return nil, "close_frame"
    end
    
    -- Handle ping frame (opcode 0x09) - respond with pong to maintain connection
    if opcode == 0x09 then
        debug_log("Received WebSocket ping frame - sending pong response", "DEBUG")
        
        -- Send pong response (opcode 0x0A) with same payload, properly masked
        local pong_opcode = 0x0A
        local pong_frame = ""
        
        -- First byte: FIN (1) + RSV (000) + Opcode (0x0A)
        pong_frame = pong_frame .. pack_u8(bit_or(0x80, pong_opcode))
        
        -- Second byte: MASK (1) + Payload length
        -- Initialize payload if not set (ping frames may have no payload)
        if not payload then
            payload = ""
        end
        local payload_len = #payload
        if payload_len < 126 then
            pong_frame = pong_frame .. pack_u8(bit_or(0x80, payload_len))
        else
            -- For simplicity, truncate large ping payloads
            payload = string.sub(payload, 1, 125)
            payload_len = 125
            pong_frame = pong_frame .. pack_u8(bit_or(0x80, payload_len))
        end
        
        -- Generate 4-byte masking key
        local mask_key = {
            math.random(0, 255),
            math.random(0, 255),
            math.random(0, 255),
            math.random(0, 255)
        }
        
        -- Add masking key to frame
        for i = 1, 4 do
            pong_frame = pong_frame .. pack_u8(mask_key[i])
        end
        
        -- Mask the payload
        local masked_payload = ""
        for i = 1, payload_len do
            local data_byte = string.byte(payload, i)
            local mask_byte = mask_key[((i - 1) % 4) + 1]
            masked_payload = masked_payload .. pack_u8(bit_xor(data_byte, mask_byte))
        end
        
        pong_frame = pong_frame .. masked_payload
        
        -- Send pong response
        local success, err = socket:send(pong_frame)
        if not success then
            debug_log("Failed to send pong response: " .. (err or "unknown error"), "ERROR")
        else
            debug_log("Pong response sent successfully", "DEBUG")
        end
        
        return nil -- Continue processing
    end
    
    -- Handle pong frame (opcode 0x0A) - just ignore
    if opcode == 0x0A then
        debug_log("Received WebSocket pong frame - ignoring", "DEBUG")
        return nil -- Continue processing
    end
    
    -- Only process text (0x01) and binary (0x02) frames
    if opcode ~= 0x01 and opcode ~= 0x02 then
        debug_log("Ignoring WebSocket frame with opcode: " .. opcode, "DEBUG")
        return nil
    end
    
    -- Handle extended payload length
    if payload_len == 126 then
        local len_data = recv_exact(socket, 2)
        if not len_data then return nil end
        -- Unpack 16-bit big-endian
        local b1, b2 = string.byte(len_data, 1, 2)
        payload_len = (b1 * 256) + b2
    elseif payload_len == 127 then
        local len_data = recv_exact(socket, 8)
        if not len_data then return nil end
        -- For simplicity, just use the lower 16 bits for 64-bit length
        local b7, b8 = string.byte(len_data, 7, 8)
        payload_len = (b7 * 256) + b8
    end
    
    -- Handle masking key (if present)
    local mask_key
    if masked then
        mask_key = recv_exact(socket, 4)
        if not mask_key then return nil end
    end
    
    -- Receive payload
    local payload, perr = recv_exact(socket, payload_len)
    if not payload then
        if perr == "closed" then
            return nil, "closed"
        end
        debug_log("Failed to receive payload of length " .. payload_len .. ": " .. (perr or "unknown"), "ERROR")
        return nil
    end
    
    -- Unmask payload if necessary
    if masked and mask_key then
        local unmasked = {}
        for i = 1, #payload do
            local mask_byte = string.byte(mask_key, ((i - 1) % 4) + 1)
            unmasked[i] = string.char(bit_xor(string.byte(payload, i), mask_byte))
        end
        payload = table.concat(unmasked)
    end
    
    return payload, opcode
end

-- ============================================================================
-- COMMUNICATION PROTOCOL IMPLEMENTATION
-- ============================================================================

-- Send initialization message
local function send_init_message()
    local init_msg = {
        type = "init",
        timestamp = now_ms(),
        fceux_version = "2.6.4", -- Adjust based on actual version
        rom_name = "Super Mario Bros (World).nes",
        protocol_version = CONFIG.PROTOCOL_VERSION
    }
    
    local json_data = json.encode(init_msg)
    return send_websocket_message(g_state.websocket, json_data, false)
end

-- Send binary game state
local function send_game_state(game_state)
    local binary_data = pack_binary_game_state(game_state)
    return send_websocket_message(g_state.websocket, binary_data, true)
end

-- Send frame advance confirmation
local function send_frame_advance()
    local frame_msg = {
        type = "frame_advance",
        frame_id = g_state.frame_id,
        timestamp = now_ms()
    }
    
    local json_data = json.encode(frame_msg)
    return send_websocket_message(g_state.websocket, json_data, false)
end

-- Send episode event
local function send_episode_event(event_type, game_state)
    local event_msg = {
        type = "episode_event",
        event = event_type,
        episode_id = g_state.episode_id,
        final_score = game_state.total_score,
        final_x_position = game_state.mario.x_pos_world,
        time_remaining = game_state.time_remaining,
        timestamp = now_ms()
    }
    
    local json_data = json.encode(event_msg)
    return send_websocket_message(g_state.websocket, json_data, false)
end

-- Send heartbeat/ping
local function send_heartbeat()
    local ping_msg = {
        type = "ping",
        timestamp = now_ms()
    }
    
    local json_data = json.encode(ping_msg)
    return send_websocket_message(g_state.websocket, json_data, false)
end

-- Send error message
local function send_error_message(error_code, message, additional_data)
    local error_msg = {
        type = "error",
        error_code = error_code,
        message = message,
        timestamp = now_ms()
    }
    
    if additional_data then
        for k, v in pairs(additional_data) do
            error_msg[k] = v
        end
    end
    
    local json_data = json.encode(error_msg)
    return send_websocket_message(g_state.websocket, json_data, false)
end

-- Execute controller input based on action
local function execute_controller_input(buttons)
    -- Set controller input using FCEUX joypad API
    joypad.set(1, buttons)
    
    if CONFIG.LOG_MEMORY_READS then
        local button_str = ""
        for button, pressed in pairs(buttons) do
            if pressed then
                button_str = button_str .. button .. " "
            end
        end
        debug_log("Controller input: " .. (button_str ~= "" and button_str or "none"))
    end
end

-- Convert action ID to controller input
local function action_to_controller_input(action_id)
    return ACTION_MAPPING[action_id] or ACTION_MAPPING[0]
end

-- Handle action message from Python trainer
local function handle_action_message(message)
    if not message.frame_id then
        debug_log("Invalid action message received - missing frame_id", "ERROR")
        return
    end
    
    -- Debug: Log the entire message to see what we're receiving
    debug_log("Received action message: " .. json.encode(message), "DEBUG")
    
    -- Check frame synchronization with tolerance for minor desyncs
    local frame_diff = math.abs(message.frame_id - g_state.frame_id)
    if frame_diff > 1 then -- Allow 1 frame difference for network delays
        debug_log(string.format("Frame desync detected: expected %d, got %d (diff: %d)",
                  g_state.frame_id, message.frame_id, frame_diff), "WARN")
        
        g_state.desync_count = g_state.desync_count + 1
        
        -- Only trigger error on severe desyncs (more than 10 frames off)
        if frame_diff > 10 then
            debug_log("Severe frame desync detected, attempting recovery", "ERROR")
            send_error_message("FRAME_DESYNC", "Severe frame ID mismatch detected", {
                expected_frame = g_state.frame_id,
                received_frame = message.frame_id,
                frame_diff = frame_diff
            })
            return
        elseif g_state.desync_count > CONFIG.MAX_FRAME_SKIP * 3 then -- Increased tolerance
            debug_log("Too many minor desyncs, attempting recovery", "ERROR")
            send_error_message("FRAME_DESYNC", "Persistent frame desyncs detected", {
                expected_frame = g_state.frame_id,
                received_frame = message.frame_id,
                desync_count = g_state.desync_count
            })
            return
        end
    else
        -- Reset desync count on successful sync
        if g_state.desync_count > 0 then
            debug_log("Frame sync recovered, resetting desync count", "INFO")
            g_state.desync_count = 0
        end
    end
    
    -- Convert action to controller input
    local buttons
    if message.action ~= nil then
        -- If we receive an action ID, convert it to button states
        debug_log("Converting action ID " .. tostring(message.action) .. " to buttons", "DEBUG")
        buttons = action_to_controller_input(message.action)
    elseif message.buttons then
        -- If we receive button states directly, use them
        debug_log("Using button states directly: " .. json.encode(message.buttons), "DEBUG")
        
        -- Ensure buttons is a table, not a string
        if type(message.buttons) == "string" then
            debug_log("ERROR: buttons field is a string, not a table. Attempting to parse...", "ERROR")
            -- Try to parse the string as JSON
            local success, parsed_buttons = pcall(json.decode, message.buttons)
            if success and type(parsed_buttons) == "table" then
                buttons = parsed_buttons
                debug_log("Successfully parsed buttons string to table", "DEBUG")
            else
                debug_log("Failed to parse buttons string, using no action", "ERROR")
                buttons = ACTION_MAPPING[0]  -- No action
            end
        else
            buttons = message.buttons
        end
    else
        debug_log("Invalid action message - missing action or buttons, using no action", "ERROR")
        buttons = ACTION_MAPPING[0]  -- No action
    end
    
    -- Final validation: ensure buttons is a table
    if type(buttons) ~= "table" then
        debug_log("ERROR: buttons is not a table (type: " .. type(buttons) .. "), using no action", "ERROR")
        buttons = ACTION_MAPPING[0]  -- No action
    end
    
    -- Debug: Log what we're about to pass to joypad.set
    debug_log("About to call joypad.set with: " .. json.encode(buttons) .. " (type: " .. type(buttons) .. ")", "DEBUG")
    
    -- Store the action to be executed in the main loop
    g_state.pending_action = buttons
    g_state.waiting_for_action = false
    
    if CONFIG.LOG_FRAME_SYNC then
        debug_log(string.format("Queued action for frame %d", message.frame_id))
    end
end

-- Game reset function to return to World 1-1
local function reset_game_to_level_1_1()
    debug_log("Resetting game to World 1-1...")
    
    -- Use the simple, reliable save state loading approach
    local savestate_loaded = false
    
    if savestate then
        -- Load save state in slot 10
        local st = savestate.object(10)
        local success = pcall(savestate.load, st)
        if success then
            savestate_loaded = true
            debug_log("Successfully loaded save state from slot 10")
        else
            debug_log("Failed to load save state from slot 10", "WARN")
        end
    else
        debug_log("savestate API not available", "ERROR")
    end
    
    if not savestate_loaded then
        debug_log("Could not load save state automatically - manual reset required", "WARN")
        debug_log("Please load the World 1-1 save state manually from slot 10", "WARN")
        debug_log("Make sure you have a save state in slot 10", "WARN")
        
        -- Send reset request to Python trainer as fallback (but don't disconnect)
        if g_state.websocket and g_state.connected then
            local reset_msg = {
                type = "reset_request",
                episode_id = g_state.episode_id,
                timestamp = now_ms(),
                manual_reset_required = true
            }
            
            local json_data = json.encode(reset_msg)
            send_websocket_message(g_state.websocket, json_data, false)
        end
    end
    
    -- Reset state tracking (but keep connection alive)
    g_state.frame_id = 0
    g_state.episode_frames = 0
    g_state.episode_start_time = now_ms()
    g_state.waiting_for_action = false
    g_state.pending_action = nil
    
    -- Reset lives tracking to prevent false death detection
    g_state.prev_lives = nil
    
    -- Reset Mario position tracking for new episode
    g_mario_true_x = 0
    g_previous_mario_x = 0
    debug_log("Reset Mario position tracking for new episode")
    
    debug_log("Game reset completed, ready for episode " .. g_state.episode_id)
end

-- Calculate reward based on Mario's progress (from old working script)
local function calculate_mario_reward(mario_x, previous_x)
    local reward = 0
    
    -- Primary reward: forward progress
    local progress = mario_x - previous_x
    if progress > 0 then
        reward = progress * 0.1  -- Small positive reward for moving right
    elseif progress < 0 then
        reward = progress * 0.2  -- Larger penalty for moving left
    end
    
    -- Target-based reward (from old script concept)
    local target_x = 120  -- Target position to encourage forward movement
    local distance_from_target = math.abs(mario_x - target_x)
    
    -- Additional reward structure
    if mario_x > previous_x then
        reward = reward + 1.0  -- Bonus for any forward movement
    end
    
    -- Penalty for going backwards significantly
    if mario_x < 107 then
        reward = reward - 10.0  -- Heavy penalty for going too far back
    end
    
    -- Bonus for reaching certain milestones
    if mario_x > 200 then
        reward = reward + 5.0   -- Bonus for significant progress
    end
    
    if mario_x > 400 then
        reward = reward + 10.0  -- Larger bonus for major progress
    end
    
    return reward
end

-- Handle training control messages
local function handle_training_control(message)
    if message.command == "start" then
        g_state.training_active = true
        g_state.episode_id = message.episode_id or g_state.episode_id
        debug_log("Training started, episode: " .. g_state.episode_id)
        
    elseif message.command == "pause" then
        g_state.training_active = false
        debug_log("Training paused")
        
    elseif message.command == "reset" then
        -- Reset game state using save state - DO NOT DISCONNECT
        debug_log("Received reset command - performing immediate save state reset")
        g_state.episode_id = message.episode_id or (g_state.episode_id + 1)
        debug_log("Starting episode: " .. g_state.episode_id)
        
        -- Perform immediate reset without disconnecting
        reset_game_to_level_1_1()
        
        -- Send episode started event to Python
        local episode_start_msg = {
            type = "episode_event",
            event = "started",
            episode_id = g_state.episode_id,
            timestamp = now_ms()
        }
        local json_data = json.encode(episode_start_msg)
        send_websocket_message(g_state.websocket, json_data, false)
        
    elseif message.command == "stop" then
        g_state.training_active = false
        debug_log("Training stopped")
    end
end

-- Process received messages
local function process_received_message(data, opcode)
    if opcode == 0x01 then -- Text frame (JSON)
        local success, message = pcall(json.decode, data)
        if not success then
            debug_log("Failed to parse JSON message: " .. data, "ERROR")
            return
        end
        
        if message.type == "init_ack" then
            debug_log("Received initialization acknowledgment")
            g_state.training_active = true
            
        elseif message.type == "action" then
            handle_action_message(message)
            
        elseif message.type == "training_control" then
            handle_training_control(message)
            
        elseif message.type == "pong" then
            -- Handle heartbeat response
            local latency = now_ms() - message.timestamp
            if CONFIG.LOG_FRAME_SYNC then
                debug_log(string.format("Heartbeat latency: %dms", latency))
            end
            
        elseif message.type == "frame_ack" then
            -- Handle frame acknowledgment
            if message.ready_for_next then
                g_state.waiting_for_action = false
            end
            
        else
            debug_log("Unknown message type: " .. (message.type or "nil"), "WARN")
        end
        
    elseif opcode == 0x02 then -- Binary frame
        debug_log("Received unexpected binary message", "WARN")
    end
end

-- ============================================================================
-- CONTROLLER INPUT EXECUTION
-- ============================================================================

-- Note: execute_controller_input and action_to_controller_input functions
-- are now defined earlier in the file before handle_action_message

-- ============================================================================
-- FRAME SYNCHRONIZATION AND TIMING
-- ============================================================================


-- Handle connection errors and attempt reconnection
local function handle_connection_error()
    debug_log("Connection error detected, attempting immediate reconnection...", "WARN")
    
    if g_state.websocket then
        g_state.websocket:close()
        g_state.websocket = nil
    end
    
    g_state.connected = false
    g_state.waiting_for_action = false
    -- Don't stop training - keep trying to reconnect
    
    -- Attempt immediate reconnection with shorter delays
    if g_state.reconnect_attempts < 10 then  -- Increased attempts
        g_state.reconnect_attempts = g_state.reconnect_attempts + 1
        
        debug_log(string.format("Reconnection attempt %d/10", g_state.reconnect_attempts))
        
        -- Much shorter wait times for faster recovery
        local wait_time = math.min(500 + (g_state.reconnect_attempts * 200), 2000)  -- Max 2 seconds
        debug_log(string.format("Waiting %dms before reconnection", wait_time))
        
        g_state.last_reconnect_attempt = now_ms()
        g_state.reconnect_delay = wait_time
        
        debug_log("Fast reconnection scheduled")
    else
        debug_log("Max reconnection attempts reached, resetting counter", "WARN")
        g_state.reconnect_attempts = 0  -- Reset and keep trying
        g_state.last_reconnect_attempt = now_ms()
        g_state.reconnect_delay = 1000  -- 1 second delay before reset
    end
end

-- Connection health check
local function check_connection_health()
    local current_time = now_ms()
    
    -- Check if we should perform health check
    if current_time - g_state.last_health_check < CONFIG.CONNECTION_HEALTH_CHECK_INTERVAL then
        return true -- Too early for health check
    end
    
    g_state.last_health_check = current_time
    
    -- Check for too many consecutive timeouts
    if g_state.consecutive_timeouts > CONFIG.MAX_CONSECUTIVE_TIMEOUTS then
        debug_log(string.format("Connection health check failed: %d consecutive timeouts",
                  g_state.consecutive_timeouts), "ERROR")
        return false
    end
    
    -- Check if we've had any successful messages recently
    if current_time - g_state.last_successful_message > 30000 then -- 30 seconds
        debug_log("Connection health check failed: no successful messages in 30 seconds", "ERROR")
        return false
    end
    
    -- Connection is healthy
    if g_state.consecutive_timeouts > 0 then
        debug_log(string.format("Connection health check passed: %d successful messages, %d timeouts",
                  g_state.successful_messages, g_state.consecutive_timeouts), "INFO")
    end
    
    return true
end

-- Process incoming WebSocket messages (drain all pending messages each frame)
local function process_incoming_messages()
    if not g_state.websocket then return end
    
    local messages_processed = 0
    local max_messages_per_frame = 10 -- Prevent infinite loops
    
    while messages_processed < max_messages_per_frame do
        local data, opcode_or_error = receive_websocket_message(g_state.websocket)
        
        if not data then
            if opcode_or_error == "timeout" then
                g_state.consecutive_timeouts = g_state.consecutive_timeouts + 1
                -- Don't treat timeouts as errors unless excessive
                break
            elseif opcode_or_error == "closed" or opcode_or_error == "close_frame" then
                debug_log("WebSocket connection closed, triggering reconnection", "WARN")
                handle_connection_error()
                return
            elseif opcode_or_error then
                debug_log("WebSocket error: " .. opcode_or_error, "WARN")
                g_state.error_count = g_state.error_count + 1
                g_state.consecutive_timeouts = g_state.consecutive_timeouts + 1
                
                -- Increased error tolerance and reset error count periodically
                if g_state.error_count > 50 then -- Further increased tolerance
                    debug_log("Too many WebSocket errors, attempting reconnection", "ERROR")
                    handle_connection_error()
                    return
                end
            end
            break -- No more messages or timeout
        else
            -- Successful message received
            g_state.consecutive_timeouts = 0
            g_state.successful_messages = g_state.successful_messages + 1
            g_state.last_successful_message = now_ms()
            
            -- Reset error count on successful message
            if g_state.error_count > 0 then
                g_state.error_count = math.max(0, g_state.error_count - 1)
            end
        end
        
        -- Process the message
        process_received_message(data, opcode_or_error)
        messages_processed = messages_processed + 1
    end
    
    if messages_processed >= max_messages_per_frame then
        debug_log("Hit max messages per frame limit: " .. max_messages_per_frame, "WARN")
    end
    
    -- Perform periodic connection health check
    if not check_connection_health() then
        debug_log("Connection health check failed, triggering reconnection", "ERROR")
        handle_connection_error()
    end
end

-- Enhanced deterministic frame processing with improved synchronization and throttling
local function process_frame()
    if not g_state.connected then
        return -- Skip if not connected
    end
    
    if not g_state.training_active then
        return -- Skip if training not active
    end
    
    -- Completely disable throttling for maximum performance
    local current_time = now_ms()
    
    -- No throttling - let the system run at full speed
    -- The non-blocking approach should handle performance naturally
    
    -- Extract current game state
    local game_state = extract_complete_game_state()
    
    -- Check for episode termination
    if game_state.is_terminal then
        local event_type = "death"
        if game_state.is_level_complete then
            event_type = "level_complete"
        elseif game_state.is_time_up then
            event_type = "time_up"
        end
        
        debug_log(string.format("Terminal condition detected: %s", event_type), "INFO")
        send_episode_event(event_type, game_state)
        
        -- Reset for next episode immediately
        g_state.episode_id = g_state.episode_id + 1
        g_state.episode_frames = 0
        g_state.episode_start_time = now_ms()
        
        -- Perform immediate reset without setting flag
        debug_log("Episode terminated, performing immediate reset", "INFO")
        reset_game_to_level_1_1()
        
        return
    end
    
    -- Send game state and wait for action (ENHANCED SYNCHRONIZATION)
    if CONFIG.LOG_FRAME_SYNC then
        debug_log(string.format("Frame %d: Sending game state (128-byte payload)", g_state.frame_id), "DEBUG")
    end
    
    if send_game_state(game_state) then
        -- NON-BLOCKING approach - don't pause emulation
        g_state.waiting_for_action = true
        g_state.pending_action = nil
        g_state.frame_timeout_start = current_time
        
        -- Very short timeout for non-blocking operation
        local timeout_ms = 16  -- Target 60 FPS = 16ms per frame
        local start_wait_time = current_time
        local max_checks = 5  -- Maximum message checks per frame
        local checks = 0
        
        -- Quick message processing without blocking
        while g_state.waiting_for_action and checks < max_checks do
            process_incoming_messages()
            checks = checks + 1
            
            -- Check if we got a response quickly
            local elapsed_time = now_ms() - start_wait_time
            if elapsed_time > timeout_ms then
                break
            end
        end
        
        -- If no response received quickly, continue anyway
        if g_state.waiting_for_action then
            debug_log(string.format("No action received within %dms, using default", timeout_ms), "DEBUG")
        end
        
        -- Always ensure we have an action - simplified logic
        if g_state.waiting_for_action or not g_state.pending_action then
            g_state.pending_action = ACTION_MAPPING[0]  -- No action
            g_state.waiting_for_action = false
            
            -- Minimal timeout tracking - don't let it build up
            g_state.consecutive_timeouts = math.min(g_state.consecutive_timeouts + 1, 10)
        else
            -- Reset timeout counter on successful action receipt
            g_state.consecutive_timeouts = 0
        end
        
        -- Execute the received action
        if g_state.pending_action then
            execute_controller_input(g_state.pending_action)
            
            if CONFIG.LOG_FRAME_SYNC then
                debug_log(string.format("Frame %d: Executed action", g_state.frame_id), "DEBUG")
            end
            
            -- Update frame counter and timing
            g_state.frame_id = g_state.frame_id + 1
            g_state.episode_frames = g_state.episode_frames + 1
            g_state.last_frame_time = current_time
            
            -- Clear action
            g_state.pending_action = nil
        end
    else
        debug_log("Failed to send game state, using default action", "ERROR")
        execute_controller_input(ACTION_MAPPING[0])  -- No action
        g_state.frame_id = g_state.frame_id + 1
        g_state.episode_frames = g_state.episode_frames + 1
        -- Don't increment timeout counter for send failures - different issue
    end
    
    -- Update tracking variables
    g_state.last_mario_x = game_state.mario.x_pos_world or 0
    g_state.last_score = game_state.total_score or 0
    
    -- Remove the frame advance throttling - it's causing the slowdown
    -- The system should run at normal speed even with some timeouts
end


-- Send periodic heartbeat
local function send_periodic_heartbeat()
    local current_time = now_ms()
    if current_time - g_state.last_heartbeat > CONFIG.HEARTBEAT_INTERVAL then
        if send_heartbeat() then
            g_state.last_heartbeat = current_time
        end
    end
end

-- ============================================================================
-- ERROR HANDLING AND RECONNECTION
-- ============================================================================


-- Validate game state for corruption
local function validate_game_state(game_state)
    local mario = game_state.mario
    
    -- Sanity checks
    if mario.x_pos_world < g_state.last_mario_x - 100 then -- Allow some backward movement
        debug_log("Mario X position validation failed", "WARN")
        return false
    end
    
    if mario.y_pos_level > 240 then -- Screen height
        debug_log("Mario Y position out of bounds", "WARN")
        return false
    end
    
    if game_state.total_score < g_state.last_score then -- Score should not decrease
        debug_log("Score validation failed", "WARN")
        return false
    end
    
    return true
end

-- Handle desync recovery
local function handle_desync_recovery()
    debug_log("Initiating desync recovery...", "WARN")
    
    -- Send resync request
    local resync_msg = {
        type = "resync_request",
        current_frame = g_state.frame_id,
        timestamp = now_ms(),
        episode_id = g_state.episode_id
    }
    
    local json_data = json.encode(resync_msg)
    if send_websocket_message(g_state.websocket, json_data, false) then
        -- Reset synchronization state but don't reset frame counter
        g_state.waiting_for_action = false
        g_state.desync_count = 0
        g_state.error_count = 0  -- Reset error count on recovery attempt
        debug_log("Desync recovery initiated, waiting for sync response")
        
        -- Wait briefly for sync response
        local sync_start = now_ms()
        while now_ms() - sync_start < 1000 do -- Wait up to 1 second
            process_incoming_messages()
            if g_state.desync_count == 0 then -- If no new desyncs, assume recovery worked
                break
            end
        end
        
        debug_log("Desync recovery completed")
    else
        debug_log("Failed to send resync request", "ERROR")
        handle_connection_error()
    end
end

-- ============================================================================
-- MAIN CONNECTION AND INITIALIZATION
-- ============================================================================

-- Connect to Python trainer
local function connect_to_trainer()
    debug_log("Connecting to Python trainer...")
    
    g_state.websocket = init_websocket()
    if not g_state.websocket then
        return false
    end
    
    g_state.connected = true
    
    -- Send initialization message
    if not send_init_message() then
        debug_log("Failed to send initialization message", "ERROR")
        g_state.websocket:close()
        g_state.websocket = nil
        g_state.connected = false
        return false
    end
    
    debug_log("Connected to trainer, waiting for acknowledgment...")
    
    -- Wait for initialization acknowledgment with better error handling
    local timeout_start = now_ms()
    local init_attempts = 0
    while now_ms() - timeout_start < 10000 do -- Increased to 10 second timeout
        local data, opcode = receive_websocket_message(g_state.websocket)
        if data and opcode == 0x01 then
            local success, message = pcall(json.decode, data)
            if success and message.type == "init_ack" then
                debug_log("Initialization acknowledged, training ready")
                g_state.training_active = true
                g_state.error_count = 0  -- Reset error count on successful init
                g_state.desync_count = 0 -- Reset desync count
                
                -- Reset connection health tracking on successful init
                g_state.consecutive_timeouts = 0
                g_state.successful_messages = 1
                g_state.last_successful_message = now_ms()
                
                return true
            end
        elseif opcode == "closed" or opcode == "close_frame" then
            debug_log("Connection closed during initialization", "ERROR")
            break
        end
        
        -- Retry sending init message every 2 seconds
        if (now_ms() - timeout_start) % 2000 < 100 and init_attempts < 3 then
            debug_log("Retrying initialization message...", "INFO")
            send_init_message()
            init_attempts = init_attempts + 1
        end
    end
    
    debug_log("Initialization timeout or connection failed", "ERROR")
    if g_state.websocket then
        g_state.websocket:close()
        g_state.websocket = nil
    end
    g_state.connected = false
    return false
end

-- Runtime self-checks for address validation
local function validate_memory_addresses()
    local debug_msg = "Memory address validation:\n"
    
    -- Test Mario position addresses
    local mario_x_page = memory.readbyte(0x006D)  -- CORRECTED: Use Mario's actual page, not camera
    local mario_x_pixel = memory.readbyte(0x0086)
    local mario_y = memory.readbyte(0x00CE)
    local player_state = memory.readbyte(0x001D)
    
    -- Test camera address
    local camera_page = memory.readbyte(0x03AD)
    
    -- Test HUD timer addresses (decimal digits, not BCD)
    local timer_hundreds = memory.readbyte(0x07F8)
    local timer_tens = memory.readbyte(0x07F9)
    local timer_ones = memory.readbyte(0x07FA)
    
    debug_msg = debug_msg .. string.format("Mario X Page (0x006D): %d\n", mario_x_page)
    debug_msg = debug_msg .. string.format("Mario X Pixel (0x0086): %d\n", mario_x_pixel)
    debug_msg = debug_msg .. string.format("Mario Y (0x00CE): %d\n", mario_y)
    debug_msg = debug_msg .. string.format("Player State (0x001D): %d\n", player_state)
    debug_msg = debug_msg .. string.format("Camera Page (0x03AD): %d\n", camera_page)
    debug_msg = debug_msg .. string.format("Timer Digits: %d%d%d\n", timer_hundreds, timer_tens, timer_ones)
    
    -- Basic sanity checks
    local valid = true
    if mario_x_page > 64 then  -- Mario shouldn't be beyond page 64 in normal gameplay (increased range)
        debug_msg = debug_msg .. "WARNING: Mario X page seems too high\n"
        valid = false
    end
    if mario_y > 240 then  -- Mario shouldn't be below screen
        debug_msg = debug_msg .. "WARNING: Mario Y position seems invalid\n"
        valid = false
    end
    if player_state > 20 then  -- Player state enum shouldn't exceed reasonable range
        debug_msg = debug_msg .. "WARNING: Player state seems invalid\n"
        valid = false
    end
    if timer_hundreds > 9 or timer_tens > 9 or timer_ones > 9 then  -- Decimal digits should be 0-9
        debug_msg = debug_msg .. "WARNING: Timer decimal values seem invalid\n"
        valid = false
    end
    
    debug_log(debug_msg)
    
    return valid
end

-- Initialize the AI training system
local function initialize_ai_system()
    -- Initialize logging first
    init_log_file()
    
    debug_log("Initializing Super Mario Bros AI Training System")
    debug_log("Protocol Version: " .. CONFIG.PROTOCOL_VERSION)
    debug_log("Logging to file: " .. (CONFIG.LOG_TO_FILE and CONFIG.LOG_FILE_PATH or "disabled"))
    
    -- Validate memory addresses
    debug_log("Validating memory addresses...")
    validate_memory_addresses()
    
    -- Initialize random seed for WebSocket masking
    math.randomseed(os.time())
    
    -- Reset global state
    g_state.frame_id = 0
    g_state.episode_id = 1
    g_state.waiting_for_action = false
    g_state.training_active = false
    g_state.connected = false
    g_state.reconnect_attempts = 0
    g_state.desync_count = 0
    g_state.error_count = 0
    g_state.last_heartbeat = now_ms()
    g_state.prev_lives = nil  -- For lives delta checking
    
    -- Reset connection health tracking
    g_state.last_health_check = now_ms()
    g_state.consecutive_timeouts = 0
    g_state.successful_messages = 0
    g_state.last_successful_message = now_ms()
    g_state.last_reconnect_attempt = 0
    g_state.reconnect_delay = 0
    
    -- Attempt initial connection
    if not connect_to_trainer() then
        debug_log("Failed to establish initial connection", "ERROR")
        return false
    end
    
    debug_log("AI Training System initialized successfully")
    return true
end

-- Cleanup function
local function cleanup_ai_system()
    debug_log("Cleaning up AI Training System...")
    
    if g_state.websocket then
        -- Send disconnect message
        local disconnect_msg = {
            type = "disconnect",
            timestamp = now_ms(),
            reason = "script_shutdown"
        }
        
        local json_data = json.encode(disconnect_msg)
        send_websocket_message(g_state.websocket, json_data, false)
        
        g_state.websocket:close()
        g_state.websocket = nil
    end
    
    g_state.connected = false
    g_state.training_active = false
    
    debug_log("Cleanup completed")
    
    -- Close log file
    close_log_file()
end

-- ============================================================================
-- MAIN EXECUTION LOOP
-- ============================================================================

-- Simplified main loop - deterministic frame-by-frame processing
local function main_loop()
    -- Handle scheduled reconnections (prevents infinite recursion)
    if not g_state.connected and g_state.reconnect_delay > 0 and g_state.last_reconnect_attempt > 0 then
        local current_time = now_ms()
        if current_time - g_state.last_reconnect_attempt >= g_state.reconnect_delay then
            debug_log("Attempting scheduled reconnection...", "INFO")
            
            -- Reset reconnection state
            g_state.last_reconnect_attempt = 0
            g_state.reconnect_delay = 0
            
            -- Try to reconnect
            if connect_to_trainer() then
                debug_log("Scheduled reconnection successful", "INFO")
                g_state.reconnect_attempts = 0
            else
                debug_log("Scheduled reconnection failed", "ERROR")
                -- Will be rescheduled by handle_connection_error if attempts remain
            end
        end
    end
    
    -- Only process if connected and training is active
    if g_state.connected and g_state.training_active then
        process_frame()
    end
    
    -- Heartbeat disabled to prevent any potential frame corruption
    -- The connection will be maintained through regular game state messages
end

-- ============================================================================
-- FCEUX INTEGRATION AND EVENT HANDLERS
-- ============================================================================

-- FCEUX event handlers
local function on_frame_start()
    main_loop()
end

local function on_frame_end()
    -- Additional frame-end processing if needed
end

local function on_reset()
    debug_log("Game reset detected")
    g_state.frame_id = 0
    g_state.episode_id = g_state.episode_id + 1
    g_state.waiting_for_action = false
end

local function on_load_state()
    debug_log("Save state loaded")
    -- Handle save state loading
end

local function on_save_state()
    debug_log("Save state created")
    -- Handle save state creation
end

-- ============================================================================
-- SCRIPT INITIALIZATION AND REGISTRATION
-- ============================================================================

-- Register FCEUX callbacks
if emu then
    -- DON'T auto-initialize - let user call connect_ai() manually
    -- This prevents FCEUX from exiting immediately
    
    -- Check if registerexit exists
    if emu.registerexit then
        emu.registerexit(function()
            debug_log("FCEUX exiting, cleaning up...")
            cleanup_ai_system()
        end)
    end
    
    -- Register frame callback - this should exist in most FCEUX versions
    if emu.registerafter then
        emu.registerafter(on_frame_start)
    elseif emu.registerframe then
        emu.registerframe(on_frame_start)
    end
    
    -- Register other callbacks if available
    if emu.registerreset then
        emu.registerreset(on_reset)
    end
    
    if emu.registerloadstate then
        emu.registerloadstate(on_load_state)
    end
    
    if emu.registersavestate then
        emu.registersavestate(on_save_state)
    end
    
    debug_log("FCEUX callbacks registered successfully")
    debug_log("=== MANUAL INITIALIZATION REQUIRED ===")
    debug_log("1. Start the Python training system first")
    debug_log("2. Then call connect_ai() in FCEUX console to begin training")
    debug_log("3. Use print_ai_status() to check connection status")
    debug_log("4. Use disconnect_ai() to stop training")
    debug_log("5. Use toggle_debug() to toggle debug output")
    debug_log("=======================================")
else
    -- If emu is not available, just log for testing
    debug_log("FCEUX emu object not available, running in test mode")
end

-- ============================================================================
-- UTILITY FUNCTIONS FOR MANUAL CONTROL
-- ============================================================================

-- Manual connection function (can be called from FCEUX console)
function connect_ai()
    return initialize_ai_system()
end

-- Manual disconnection function
function disconnect_ai()
    cleanup_ai_system()
end

-- Get current status
function get_ai_status()
    return {
        connected = g_state.connected,
        training_active = g_state.training_active,
        frame_id = g_state.frame_id,
        episode_id = g_state.episode_id,
        waiting_for_action = g_state.waiting_for_action,
        error_count = g_state.error_count,
        desync_count = g_state.desync_count
    }
end

-- Print status to console
function print_ai_status()
    local status = get_ai_status()
    print("=== AI Training System Status ===")
    print("Connected: " .. tostring(status.connected))
    print("Training Active: " .. tostring(status.training_active))
    print("Frame ID: " .. status.frame_id)
    print("Episode ID: " .. status.episode_id)
    print("Waiting for Action: " .. tostring(status.waiting_for_action))
    print("Error Count: " .. status.error_count)
    print("Desync Count: " .. status.desync_count)
    print("================================")
end

-- Toggle debug logging
function toggle_debug()
    CONFIG.DEBUG_ENABLED = not CONFIG.DEBUG_ENABLED
    debug_log("Debug logging " .. (CONFIG.DEBUG_ENABLED and "enabled" or "disabled"))
end

-- Manual log file functions
function open_log_file()
    init_log_file()
    debug_log("Log file opened manually")
end

function close_log_file_manual()
    debug_log("Closing log file manually")
    close_log_file()
end

function rotate_log_file_manual()
    debug_log("Rotating log file manually")
    rotate_log_file()
end

-- ============================================================================
-- SCRIPT ENTRY POINT
-- ============================================================================

-- Auto-initialize if running in FCEUX
if emu then
    debug_log("Super Mario Bros AI Training System loaded")
    debug_log("Auto-initializing connection (no manual console available)")
    
    -- Auto-initialize the AI system since FCEUX doesn't have a console
    local init_success = initialize_ai_system()
    if init_success then
        debug_log("✅ Auto-initialization successful - training ready!")
    else
        debug_log("❌ Auto-initialization failed - check Python server")
        debug_log("Make sure the Python training system is running first")
    end
else
    print("Warning: Not running in FCEUX environment")
end

-- NEW: Enhanced feature control functions
function toggle_enhanced_features()
    CONFIG.ENHANCED_MEMORY_ENABLED = not CONFIG.ENHANCED_MEMORY_ENABLED
    debug_log("Enhanced memory features " .. (CONFIG.ENHANCED_MEMORY_ENABLED and "enabled" or "disabled"))
end

function toggle_enemy_detection()
    CONFIG.ENEMY_DETECTION_ENABLED = not CONFIG.ENEMY_DETECTION_ENABLED
    debug_log("Enemy detection " .. (CONFIG.ENEMY_DETECTION_ENABLED and "enabled" or "disabled"))
end

function toggle_powerup_detection()
    CONFIG.POWERUP_DETECTION_ENABLED = not CONFIG.POWERUP_DETECTION_ENABLED
    debug_log("Power-up detection " .. (CONFIG.POWERUP_DETECTION_ENABLED and "enabled" or "disabled"))
end

function toggle_tile_sampling()
    CONFIG.TILE_SAMPLING_ENABLED = not CONFIG.TILE_SAMPLING_ENABLED
    debug_log("Level tile sampling " .. (CONFIG.TILE_SAMPLING_ENABLED and "enabled" or "disabled"))
end

function toggle_enhanced_death_detection()
    CONFIG.ENHANCED_DEATH_DETECTION = not CONFIG.ENHANCED_DEATH_DETECTION
    debug_log("Enhanced death detection " .. (CONFIG.ENHANCED_DEATH_DETECTION and "enabled" or "disabled"))
end

function toggle_velocity_tracking()
    CONFIG.VELOCITY_TRACKING_ENABLED = not CONFIG.VELOCITY_TRACKING_ENABLED
    debug_log("Velocity tracking " .. (CONFIG.VELOCITY_TRACKING_ENABLED and "enabled" or "disabled"))
end

-- Enhanced status function with new features
function print_enhanced_status()
    print("=== Enhanced Features Status ===")
    print("Enhanced Memory: " .. tostring(CONFIG.ENHANCED_MEMORY_ENABLED))
    print("Enemy Detection: " .. tostring(CONFIG.ENEMY_DETECTION_ENABLED))
    print("Power-up Detection: " .. tostring(CONFIG.POWERUP_DETECTION_ENABLED))
    print("Tile Sampling: " .. tostring(CONFIG.TILE_SAMPLING_ENABLED))
    print("Enhanced Death Detection: " .. tostring(CONFIG.ENHANCED_DEATH_DETECTION))
    print("Velocity Tracking: " .. tostring(CONFIG.VELOCITY_TRACKING_ENABLED))
    print("===============================")
end

-- Memory address validation function
function validate_enhanced_addresses()
    debug_log("Validating enhanced memory addresses...")
    
    -- Test enhanced Mario addresses
    local mario_x_page = memory.readbyte(MEMORY_ADDRESSES.MARIO_X_PAGE)
    local mario_x_sub = memory.readbyte(MEMORY_ADDRESSES.MARIO_X_SUB)
    local mario_velocity_x = memory.readbyte(MEMORY_ADDRESSES.MARIO_VELOCITY_X)
    local mario_velocity_y = memory.readbyte(MEMORY_ADDRESSES.MARIO_VELOCITY_Y)
    
    debug_log(string.format("Mario enhanced data: page=%d, sub=%d, vel_x=%d, vel_y=%d",
              mario_x_page, mario_x_sub, mario_velocity_x, mario_velocity_y))
    
    -- Test enemy addresses
    for i = 1, 5 do
        local enemy_type = memory.readbyte(MEMORY_ADDRESSES.ENEMY_SLOTS[i])
        local enemy_x_page = memory.readbyte(MEMORY_ADDRESSES.ENEMY_X_PAGE[i])
        local enemy_state = memory.readbyte(MEMORY_ADDRESSES.ENEMY_STATE[i])
        
        if enemy_type > 0 then
            debug_log(string.format("Enemy %d: type=%d, x_page=%d, state=%d",
                      i, enemy_type, enemy_x_page, enemy_state))
        end
    end
    
    -- Test power-up addresses
    local powerup_type = memory.readbyte(MEMORY_ADDRESSES.POWERUP_TYPE)
    if powerup_type > 0 then
        debug_log(string.format("Power-up detected: type=%d", powerup_type))
    end
    
    debug_log("Enhanced address validation completed")
end

-- End of script
debug_log("mario_ai.lua script loaded successfully")
debug_log("Enhanced memory reading system integrated")
if emu then
    debug_log("Auto-initialization enabled for testing")
    debug_log("Enhanced features available: toggle_enhanced_features(), print_enhanced_status(), validate_enhanced_addresses()")
else
    debug_log("Manual functions available: connect_ai(), disconnect_ai(), print_ai_status()")
    debug_log("Enhanced functions: toggle_enhanced_features(), print_enhanced_status(), validate_enhanced_addresses()")
end