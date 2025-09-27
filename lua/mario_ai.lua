
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
    FRAME_TIMEOUT_MS = 100,
    MAX_FRAME_SKIP = 2,
    HEARTBEAT_INTERVAL = 1000, -- ms
    
    -- Reconnection settings
    MAX_RECONNECT_ATTEMPTS = 3,
    RECONNECT_DELAY_MS = 1000,
    
    -- Debug settings
    DEBUG_ENABLED = true,
    LOG_MEMORY_READS = false,
    LOG_FRAME_SYNC = true
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
    
    -- Frame synchronization
    frame_id = 0,
    waiting_for_action = false,
    last_frame_time = 0,
    frame_timeout_start = 0,
    
    -- Game state
    episode_id = 0,
    training_active = false,
    last_mario_x = 0,
    last_score = 0,
    
    -- Error tracking
    desync_count = 0,
    error_count = 0,
    last_error_time = 0
}

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

-- Get current timestamp in milliseconds
local function get_timestamp_ms()
    return math.floor(os.clock() * 1000)
end

-- Debug logging function
local function debug_log(message, level)
    if not CONFIG.DEBUG_ENABLED then return end
    
    level = level or "INFO"
    local timestamp = get_timestamp_ms()
    local log_message = string.format("[%d] [%s] %s", timestamp, level, message)
    print(log_message)
    
    -- Also write to FCEUX console
    if emu and emu.print then
        emu.print(log_message)
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

-- Read Mario's core state information
local function read_mario_state()
    local mario_data = {
        -- Position and movement
        x_pos_screen = memory.readbyte(0x006D),
        x_pos_level = memory.readbyte(0x0086),
        x_pos_level_high = memory.readbyte(0x03AD),
        y_pos_screen = memory.readbyte(0x00CE),
        y_pos_level = memory.readbyte(0x03B8),
        x_velocity = signed_byte(memory.readbyte(0x0057)),
        y_velocity = signed_byte(memory.readbyte(0x009F)),
        direction = memory.readbyte(0x0045),
        on_ground = memory.readbyte(0x001D),
        
        -- Power state and status
        power_state = memory.readbyte(0x0756),
        lives = memory.readbyte(0x075A),
        invincibility_timer = memory.readbyte(0x079E),
        star_power_timer = memory.readbyte(0x0770),
        animation_state = memory.readbyte(0x0079),
        crouching = memory.readbyte(0x001E)
    }
    
    -- Calculate world position
    mario_data.x_pos_world = mario_data.x_pos_level + (mario_data.x_pos_level_high * 256)
    
    return mario_data
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
        screen_y = memory.readbyte(0x00B5),
        vertical_scroll = memory.readbyte(0x0725)
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
        oneup_flag = memory.readbyte(0x0772)
    }
end

-- Read enemy information (8 enemy slots)
local function read_enemy_info()
    local enemies = {}
    
    for i = 0, 7 do
        enemies[i] = {
            x_pos = memory.readbyte(0x0087 + i),
            y_pos = memory.readbyte(0x00CF + i),
            type = memory.readbyte(0x0014 + i),
            state = memory.readbyte(0x001C + i),
            x_velocity = signed_byte(memory.readbyte(0x0058 + i)),
            y_velocity = signed_byte(memory.readbyte(0x00A0 + i)),
            direction = memory.readbyte(0x0046 + i)
        }
    end
    
    return enemies
end

-- Read game state and control information
local function read_game_state()
    return {
        game_engine_state = memory.readbyte(0x0770),
        player_state = memory.readbyte(0x001D),
        game_mode = memory.readbyte(0x000E),
        end_of_level_flag = memory.readbyte(0x0772),
        controller_input = memory.readbyte(0x00F7),
        controller_previous = memory.readbyte(0x00F6)
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

-- Comprehensive game state extraction
local function extract_complete_game_state()
    local mario = read_mario_state()
    local level = read_level_info()
    local score = read_score_info()
    local enemies = read_enemy_info()
    local game = read_game_state()
    local objects = read_objects_info()
    
    -- Calculate derived values
    local total_score = (score.score_100k * 100000) + (score.score_10k * 10000) + 
                       (score.score_1k * 1000) + (score.score_100 * 100) + 
                       (score.score_10 * 10) + score.score_1
    
    local total_coins = (score.coins_tens * 10) + score.coins_ones
    
    local time_remaining = (level.timer_hundreds * 100) + (level.timer_tens * 10) + level.timer_ones
    
    local level_progress = mario.x_pos_world / 3168.0 -- World 1-1 length
    
    -- Check for terminal conditions
    local is_dead = mario.lives == 0 or game.player_state == 0x0B
    local is_level_complete = game.end_of_level_flag == 1
    local is_time_up = time_remaining == 0
    
    return {
        -- Frame metadata
        frame_id = g_state.frame_id,
        timestamp = get_timestamp_ms(),
        episode_id = g_state.episode_id,
        
        -- Mario state
        mario = mario,
        
        -- Level information
        level = level,
        
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
        is_dead = is_dead,
        is_level_complete = is_level_complete,
        is_time_up = is_time_up,
        is_terminal = is_dead or is_level_complete or is_time_up
    }
end

-- ============================================================================
-- BINARY DATA ENCODING
-- ============================================================================

-- Pack game state into binary format for efficient transmission
local function pack_binary_game_state(game_state)
    local mario = game_state.mario
    local level = game_state.level
    local score = game_state.score
    
    -- Mario Data Block (16 bytes)
    local mario_data = string.pack("<I2I2i1i1BBBBBBBB",
        mario.x_pos_world,      -- 2 bytes: X Position (world coordinates)
        mario.y_pos_level,      -- 2 bytes: Y Position (world coordinates)
        mario.x_velocity,       -- 1 byte: X Velocity (signed)
        mario.y_velocity,       -- 1 byte: Y Velocity (signed)
        mario.power_state,      -- 1 byte: Power State
        mario.animation_state,  -- 1 byte: Animation State
        mario.direction,        -- 1 byte: Direction Facing
        mario.on_ground,        -- 1 byte: On Ground Flag
        mario.lives,            -- 1 byte: Lives Remaining
        mario.invincibility_timer, -- 1 byte: Invincibility Timer
        0,                      -- 1 byte: Reserved
        0                       -- 1 byte: Reserved
    )
    
    -- Enemy Data Block (32 bytes, up to 8 enemies)
    local enemy_data = ""
    for i = 0, 7 do
        local enemy = game_state.enemies[i]
        enemy_data = enemy_data .. string.pack("BBBB",
            enemy.type,         -- 1 byte: Enemy Type
            enemy.x_pos,        -- 1 byte: X Position (screen relative)
            enemy.y_pos,        -- 1 byte: Y Position (screen relative)
            enemy.state         -- 1 byte: State Flags
        )
    end
    
    -- Level Data Block (64 bytes)
    local level_data = string.pack("<I2BBBBBBI2I2",
        level.screen_x_low + (level.screen_x_high * 256), -- 2 bytes: Camera X Position
        level.world_number,     -- 1 byte: World Number
        level.level_number,     -- 1 byte: Level Number
        score.score_100k,       -- 1 byte: Score (100,000s)
        score.score_10k,        -- 1 byte: Score (10,000s)
        score.score_1k,         -- 1 byte: Score (1,000s)
        score.score_100,        -- 1 byte: Score (100s)
        game_state.time_remaining, -- 4 bytes: Time Remaining
        game_state.total_coins  -- 2 bytes: Coins Collected
    )
    -- Pad level data to 64 bytes
    level_data = level_data .. string.rep("\0", 64 - #level_data)
    
    -- Game Variables Block (16 bytes)
    local game_data = string.pack("<BBHI4I4I4",
        game_state.game.game_engine_state, -- 1 byte: Game State
        math.floor(game_state.level_progress * 100), -- 1 byte: Level Progress Percentage
        3168 - mario.x_pos_world, -- 2 bytes: Distance to Flag
        game_state.frame_id,    -- 4 bytes: Frame Counter
        get_timestamp_ms(),     -- 4 bytes: Episode Timer
        0                       -- 4 bytes: Reserved
    )
    
    -- Create header (8 bytes)
    local payload = mario_data .. enemy_data .. level_data .. game_data
    local header = string.pack("<BBI4BI1",
        0x01,                   -- 1 byte: Message Type (0x01 = game_state)
        game_state.frame_id,    -- 4 bytes: Frame ID
        #payload,               -- 2 bytes: Data Length
        0                       -- 1 byte: Checksum (simple XOR, calculated below)
    )
    
    -- Calculate simple XOR checksum
    local checksum = 0
    for i = 1, #payload do
        checksum = checksum ~ string.byte(payload, i)
    end
    header = string.pack("<BBI4BI1",
        0x01, game_state.frame_id, #payload, checksum
    )
    
    return header .. payload
end

-- ============================================================================
-- WEBSOCKET COMMUNICATION (SIMPLIFIED IMPLEMENTATION)
-- ============================================================================

-- Note: This is a simplified WebSocket implementation for FCEUX Lua
-- In a real implementation, you would need a proper WebSocket library

local socket = require("socket")
local json = require("json") -- Assuming a JSON library is available

-- Initialize WebSocket connection
local function init_websocket()
    debug_log("Initializing WebSocket connection...")
    
    local tcp_socket = socket.tcp()
    tcp_socket:settimeout(5) -- 5 second timeout
    
    local result, err = tcp_socket:connect(CONFIG.WEBSOCKET_HOST, CONFIG.WEBSOCKET_PORT)
    if not result then
        debug_log("Failed to connect to WebSocket server: " .. (err or "unknown error"), "ERROR")
        return nil
    end
    
    -- Perform WebSocket handshake (simplified)
    local handshake = string.format(
        "GET / HTTP/1.1\r\n" ..
        "Host: %s:%d\r\n" ..
        "Upgrade: websocket\r\n" ..
        "Connection: Upgrade\r\n" ..
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n" ..
        "Sec-WebSocket-Version: 13\r\n" ..
        "\r\n",
        CONFIG.WEBSOCKET_HOST, CONFIG.WEBSOCKET_PORT
    )
    
    tcp_socket:send(handshake)
    
    -- Read handshake response (simplified)
    local response = tcp_socket:receive("*l")
    if not response or not string.find(response, "101 Switching Protocols") then
        debug_log("WebSocket handshake failed", "ERROR")
        tcp_socket:close()
        return nil
    end
    
    -- Skip remaining headers
    repeat
        local line = tcp_socket:receive("*l")
    until not line or line == ""
    
    tcp_socket:settimeout(0.001) -- Non-blocking for game loop
    
    debug_log("WebSocket connection established")
    return tcp_socket
end

-- Send WebSocket message (simplified frame format)
local function send_websocket_message(socket, data, is_binary)
    if not socket then return false end
    
    local opcode = is_binary and 0x02 or 0x01
    local payload_len = #data
    local frame
    
    if payload_len < 126 then
        frame = string.pack("BB", 0x80 | opcode, payload_len) .. data
    elseif payload_len < 65536 then
        frame = string.pack("BBI2", 0x80 | opcode, 126, payload_len) .. data
    else
        frame = string.pack("BBI8", 0x80 | opcode, 127, payload_len) .. data
    end
    
    local result, err = socket:send(frame)
    if not result then
        debug_log("Failed to send WebSocket message: " .. (err or "unknown error"), "ERROR")
        return false
    end
    
    return true
end

-- Receive WebSocket message (simplified)
local function receive_websocket_message(socket)
    if not socket then return nil end
    
    local header, err = socket:receive(2)
    if not header then
        if err ~= "timeout" then
            debug_log("WebSocket receive error: " .. (err or "unknown error"), "ERROR")
        end
        return nil
    end
    
    local first_byte, second_byte = string.byte(header, 1, 2)
    local fin = (first_byte & 0x80) ~= 0
    local opcode = first_byte & 0x0F
    local masked = (second_byte & 0x80) ~= 0
    local payload_len = second_byte & 0x7F
    
    -- Handle extended payload length
    if payload_len == 126 then
        local len_data = socket:receive(2)
        if not len_data then return nil end
        payload_len = string.unpack(">I2", len_data)
    elseif payload_len == 127 then
        local len_data = socket:receive(8)
        if not len_data then return nil end
        payload_len = string.unpack(">I8", len_data)
    end
    
    -- Handle masking key (if present)
    local mask_key
    if masked then
        mask_key = socket:receive(4)
        if not mask_key then return nil end
    end
    
    -- Receive payload
    local payload = socket:receive(payload_len)
    if not payload then return nil end
    
    -- Unmask payload if necessary
    if masked and mask_key then
        local unmasked = {}
        for i = 1, #payload do
            local mask_byte = string.byte(mask_key, ((i - 1) % 4) + 1)
            unmasked[i] = string.char(string.byte(payload, i) ~ mask_byte)
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
        timestamp = get_timestamp_ms(),
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
        timestamp = get_timestamp_ms()
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
        timestamp = get_timestamp_ms()
    }
    
    local json_data = json.encode(event_msg)
    return send_websocket_message(g_state.websocket, json_data, false)
end

-- Send heartbeat/ping
local function send_heartbeat()
    local ping_msg = {
        type = "ping",
        timestamp = get_timestamp_ms()
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
        timestamp = get_timestamp_ms()
    }
    
    if additional_data then
        for k, v in pairs(additional_data) do
            error_msg[k] = v
        end
    end
    
    local json_data = json.encode(error_msg)
    return send_websocket_message(g_state.websocket, json_data, false)
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
            local latency = get_timestamp_ms() - message.timestamp
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

-- Handle action message from Python trainer
local function handle_action_message(message)
    if not message.frame_id or not message.buttons then
        debug_log("Invalid action message received", "ERROR")
        return
    end
    
    -- Check frame synchronization
    if message.frame_id ~= g_state.frame_id then
        debug_log(string.format("Frame desync detected: expected %d, got %d", 
                  g_state.frame_id, message.frame_id), "ERROR")
        
        g_state.desync_count = g_state.desync_count + 1
        
        if g_state.desync_count > CONFIG.MAX_FRAME_SKIP then
            send_error_message("FRAME_DESYNC", "Frame ID mismatch detected", {
                expected_frame = g_state.frame_id,
                received_frame = message.frame_id
            })
            return
        end
    end
    
    -- Execute controller input
    execute_controller_input(message.buttons)
    
    -- Advance frame and update state
    emu.frameadvance()
    g_state.frame_id = g_state.frame_id + 1
    g_state.waiting_for_action = false
    
    -- Send frame advance confirmation
    send_frame_advance()
    
    if CONFIG.LOG_FRAME_SYNC then
        debug_log(string.format("Executed action for frame %d", message.frame_id))
    end
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
        -- Reset game state
        emu.poweron() -- Reset the emulator
        g_state.frame_id = 0
        g_state.episode_id = message.episode_id or (g_state.episode_id + 1)
        g_state.waiting_for_action = false
        debug_log("Game reset, new episode: " .. g_state.episode_id)
        
    elseif message.command == "stop" then
        g_state.training_active = false
        debug_log("Training stopped")
    end
end

-- ============================================================================
-- CONTROLLER INPUT EXECUTION
-- ============================================================================

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

-- ============================================================================
-- FRAME SYNCHRONIZATION AND TIMING
-- ============================================================================

-- Main frame processing function
local function process_frame()
    if not g_state.connected or not g_state.training_active then
        return
    end
    
    -- Check for timeout
    if g_state.waiting_for_action then
        local current_time = get_timestamp_ms()
        if current_time - g_state.frame_timeout_start > CONFIG.FRAME_TIMEOUT_MS then
            debug_log("Frame timeout detected", "WARN")
            g_state.waiting_for_action = false
            g_state.error_count = g_state.error_count + 1
        else
            return -- Still waiting for action
        end
    end
    
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
        
        send_episode_event(event_type, game_state)
        g_state.episode_id = g_state.episode_id + 1
        debug_log("Episode ended: " .. event_type)
    end
    
    -- Send game state to Python trainer
    if send_game_state(game_state) then
        g_state.waiting_for_action = true
        g_state.frame_timeout_start = get_timestamp_ms()
        
        if CONFIG.LOG_FRAME_SYNC then
            debug_log(string.format("Sent game state for frame %d", g_state.frame_id))
        end
    else
        debug_log("Failed to send game state", "ERROR")
        g_state.error_count = g_state.error_count + 1
    end
    
    -- Update tracking variables
    g_state.last_mario_x = game_state.mario.x_pos_world
    g_state.last_score = game_state.total_score
end

-- Process incoming WebSocket messages
local function process_incoming_messages()
    if not g_state.websocket then return end
    
    local data, opcode = receive_websocket_message(g_state.websocket)
    if data then
        process_received_message(data, opcode)
    end
end

-- Send periodic heartbeat
local function send_periodic_heartbeat()
    local current_time = get_timestamp_ms()
    if current_time - g_state.last_heartbeat > CONFIG.HEARTBEAT_INTERVAL then
        if send_heartbeat() then
            g_state.last_heartbeat = current_time
        end
    end
end

-- ============================================================================
-- ERROR HANDLING AND RECONNECTION
-- ============================================================================

-- Handle connection errors and attempt reconnection
local function handle_connection_error()
    debug_log("Connection error detected, attempting reconnection...", "WARN")
    
    if g_state.websocket then
        g_state.websocket:close()
        g_state.websocket = nil
    end
    
    g_state.connected = false
    g_state.training_active = false
    g_state.waiting_for_action = false
    
    -- Attempt reconnection
    if g_state.reconnect_attempts < CONFIG.MAX_RECONNECT_ATTEMPTS then
        g_state.reconnect_attempts = g_state.reconnect_attempts + 1
        
        debug_log(string.format("Reconnection attempt %d/%d",
                  g_state.reconnect_attempts, CONFIG.MAX_RECONNECT_ATTEMPTS))
        
        -- Wait before reconnecting
        local start_time = get_timestamp_ms()
        while get_timestamp_ms() - start_time < CONFIG.RECONNECT_DELAY_MS do
            -- Busy wait (Lua doesn't have sleep in FCEUX)
        end
        
        -- Attempt to reconnect
        if connect_to_trainer() then
            g_state.reconnect_attempts = 0
            debug_log("Reconnection successful")
        end
    else
        debug_log("Max reconnection attempts reached, giving up", "ERROR")
        send_error_message("CONNECTION_FAILED", "Unable to reconnect to trainer")
    end
end

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
        timestamp = get_timestamp_ms()
    }
    
    local json_data = json.encode(resync_msg)
    if send_websocket_message(g_state.websocket, json_data, false) then
        -- Reset synchronization state
        g_state.waiting_for_action = false
        g_state.desync_count = 0
        debug_log("Desync recovery initiated")
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
    
    -- Wait for initialization acknowledgment
    local timeout_start = get_timestamp_ms()
    while get_timestamp_ms() - timeout_start < 5000 do -- 5 second timeout
        local data, opcode = receive_websocket_message(g_state.websocket)
        if data and opcode == 0x01 then
            local success, message = pcall(json.decode, data)
            if success and message.type == "init_ack" then
                debug_log("Initialization acknowledged, training ready")
                g_state.training_active = true
                return true
            end
        end
    end
    
    debug_log("Initialization timeout", "ERROR")
    g_state.websocket:close()
    g_state.websocket = nil
    g_state.connected = false
    return false
end

-- Initialize the AI training system
local function initialize_ai_system()
    debug_log("Initializing Super Mario Bros AI Training System")
    debug_log("Protocol Version: " .. CONFIG.PROTOCOL_VERSION)
    
    -- Reset global state
    g_state.frame_id = 0
    g_state.episode_id = 1
    g_state.waiting_for_action = false
    g_state.training_active = false
    g_state.connected = false
    g_state.reconnect_attempts = 0
    g_state.desync_count = 0
    g_state.error_count = 0
    g_state.last_heartbeat = get_timestamp_ms()
    
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
            timestamp = get_timestamp_ms(),
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
end

-- ============================================================================
-- MAIN EXECUTION LOOP
-- ============================================================================

-- Main loop function called every frame by FCEUX
local function main_loop()
    -- Handle connection issues
    if g_state.connected and not g_state.websocket then
        handle_connection_error()
        return
    end
    
    -- Process incoming messages
    if g_state.connected then
        process_incoming_messages()
    end
    
    -- Send periodic heartbeat
    if g_state.connected then
        send_periodic_heartbeat()
    end
    
    -- Process game frame if training is active
    if g_state.training_active then
        process_frame()
    end
    
    -- Handle excessive errors
    if g_state.error_count > 10 then
        debug_log("Too many errors, resetting connection", "ERROR")
        handle_connection_error()
        g_state.error_count = 0
    end
    
    -- Handle excessive desyncs
    if g_state.desync_count > CONFIG.MAX_FRAME_SKIP then
        handle_desync_recovery()
    end
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
    emu.registerstart(function()
        debug_log("FCEUX started, initializing AI system...")
        initialize_ai_system()
    end)
    
    emu.registerexit(function()
        debug_log("FCEUX exiting, cleaning up...")
        cleanup_ai_system()
    end)
    
    -- Register frame callback
    emu.registerafter(on_frame_start)
    
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

-- ============================================================================
-- SCRIPT ENTRY POINT
-- ============================================================================

-- Auto-initialize if running in FCEUX
if emu then
    debug_log("Super Mario Bros AI Training System loaded")
    debug_log("Use connect_ai() to start training")
    debug_log("Use print_ai_status() to check status")
    debug_log("Use toggle_debug() to toggle debug output")
else
    print("Warning: Not running in FCEUX environment")
end

-- End of script
debug_log("mario_ai.lua script loaded successfully")