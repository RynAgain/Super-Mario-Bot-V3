--[[
Super Mario Bros Memory Address Verification Script
This script helps verify the correct memory addresses for reading game state
]]

-- Common Super Mario Bros memory addresses (multiple ROM versions)
local MEMORY_ADDRESSES = {
    -- Mario Position (most common addresses)
    mario_x_screen = {0x006D, 0x0086, 0x03AD},  -- Screen X, Level X low, Level X high
    mario_y_screen = {0x00CE, 0x03B8},          -- Screen Y, Level Y
    mario_x_velocity = {0x0057},
    mario_y_velocity = {0x009F},
    
    -- Mario State
    mario_lives = {0x075A, 0x075B},             -- Lives remaining
    mario_power = {0x0756, 0x0757},             -- Power state
    mario_direction = {0x0045},                 -- Direction facing
    mario_on_ground = {0x001D},                 -- On ground flag
    
    -- Game State
    game_mode = {0x000E, 0x0770},               -- Game mode
    player_state = {0x001D},                    -- Player state
    end_level_flag = {0x0772},                  -- End of level flag
    
    -- Score and Timer
    score_digits = {0x07DD, 0x07DE, 0x07DF, 0x07E0, 0x07E1, 0x07E2}, -- Score digits
    timer_digits = {0x071A, 0x071B, 0x071C},   -- Timer digits
    coins = {0x075D, 0x075E},                   -- Coins (ones, tens)
}

-- Function to test memory addresses
local function test_memory_addresses()
    print("=== Super Mario Bros Memory Address Test ===")
    print("Move Mario around and watch these values change:")
    print()
    
    -- Test Mario position addresses
    print("Mario Position Addresses:")
    for _, addr in ipairs(MEMORY_ADDRESSES.mario_x_screen) do
        local value = memory.readbyte(addr)
        print(string.format("  0x%04X: %3d (0x%02X)", addr, value, value))
    end
    
    for _, addr in ipairs(MEMORY_ADDRESSES.mario_y_screen) do
        local value = memory.readbyte(addr)
        print(string.format("  0x%04X: %3d (0x%02X)", addr, value, value))
    end
    print()
    
    -- Test Mario state
    print("Mario State:")
    for _, addr in ipairs(MEMORY_ADDRESSES.mario_lives) do
        local value = memory.readbyte(addr)
        print(string.format("  Lives 0x%04X: %3d", addr, value))
    end
    
    for _, addr in ipairs(MEMORY_ADDRESSES.mario_power) do
        local value = memory.readbyte(addr)
        print(string.format("  Power 0x%04X: %3d", addr, value))
    end
    print()
    
    -- Test score
    print("Score Digits:")
    for i, addr in ipairs(MEMORY_ADDRESSES.score_digits) do
        local value = memory.readbyte(addr)
        print(string.format("  Score[%d] 0x%04X: %3d", i, addr, value))
    end
    print()
    
    -- Test timer
    print("Timer Digits:")
    for i, addr in ipairs(MEMORY_ADDRESSES.timer_digits) do
        local value = memory.readbyte(addr)
        print(string.format("  Timer[%d] 0x%04X: %3d", i, addr, value))
    end
    print()
    
    -- Calculate world position using different methods
    print("Position Calculations:")
    local screen_x = memory.readbyte(0x006D)
    local level_x_low = memory.readbyte(0x0086)
    local level_x_high = memory.readbyte(0x03AD)
    local world_x = level_x_low + (level_x_high * 256)
    
    print(string.format("  Screen X: %d", screen_x))
    print(string.format("  Level X Low: %d", level_x_low))
    print(string.format("  Level X High: %d", level_x_high))
    print(string.format("  Calculated World X: %d", world_x))
    print()
    
    -- Alternative position calculation
    local alt_x = memory.readbyte(0x0086) + (memory.readbyte(0x03AD) * 256)
    print(string.format("  Alternative World X: %d", alt_x))
    print()
    
    print("=== End Test ===")
    print()
end

-- Function to continuously monitor key addresses
local function monitor_addresses()
    local last_values = {}
    
    -- Key addresses to monitor
    local key_addresses = {
        {0x006D, "Screen X"},
        {0x0086, "Level X Low"},
        {0x03AD, "Level X High"},
        {0x00CE, "Screen Y"},
        {0x03B8, "Level Y"},
        {0x075A, "Lives"},
        {0x0756, "Power"},
        {0x071A, "Timer 100s"},
        {0x071B, "Timer 10s"},
        {0x071C, "Timer 1s"},
    }
    
    for _, addr_info in ipairs(key_addresses) do
        local addr = addr_info[1]
        local name = addr_info[2]
        local value = memory.readbyte(addr)
        
        if last_values[addr] ~= value then
            print(string.format("%s (0x%04X): %d -> %d", name, addr, last_values[addr] or 0, value))
            last_values[addr] = value
        end
    end
end

-- Manual test functions
function test_memory()
    test_memory_addresses()
end

function monitor_memory()
    monitor_addresses()
end

-- Auto-run test
print("Memory test script loaded!")
print("Use test_memory() to run a one-time test")
print("Use monitor_memory() to continuously monitor changes")
print("Move Mario around to see which addresses change correctly")

-- Run initial test
test_memory_addresses()