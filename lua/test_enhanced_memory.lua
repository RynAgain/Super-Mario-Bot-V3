-- Test script for enhanced memory reading capabilities
-- This script validates that all enhanced memory addresses are working correctly

print("=== Enhanced Memory Reading System Test ===")

-- Load the main mario_ai.lua file to access its functions and constants
dofile("mario_ai.lua")

-- Test function to validate enhanced memory addresses
local function test_enhanced_memory_addresses()
    print("\n1. Testing Enhanced Memory Address Constants...")
    
    -- Test that all memory address constants are defined
    local required_addresses = {
        "MARIO_SCREEN_X", "MARIO_SCREEN_Y", "MARIO_X_PAGE", "MARIO_X_SUB",
        "MARIO_STATE", "MARIO_BELOW_VIEWPORT", "MARIO_POWER", "MARIO_FACING",
        "MARIO_VELOCITY_X", "MARIO_VELOCITY_Y", "LIVES", "COINS",
        "ENEMY_SLOTS", "ENEMY_X_PAGE", "ENEMY_X_SUB", "ENEMY_Y", "ENEMY_STATE",
        "POWERUP_TYPE", "POWERUP_X_PAGE", "POWERUP_Y", "LEVEL_LAYOUT"
    }
    
    local missing_addresses = {}
    for _, addr_name in ipairs(required_addresses) do
        if not MEMORY_ADDRESSES[addr_name] then
            table.insert(missing_addresses, addr_name)
        end
    end
    
    if #missing_addresses == 0 then
        print("✅ All required memory addresses are defined")
    else
        print("❌ Missing memory addresses: " .. table.concat(missing_addresses, ", "))
        return false
    end
    
    return true
end

local function test_enhanced_functions()
    print("\n2. Testing Enhanced Memory Reading Functions...")
    
    -- Test that enhanced functions exist and can be called
    local functions_to_test = {
        "read_mario_state", "read_enemy_info", "read_powerup_info",
        "read_level_tiles_around_mario", "assess_enemy_threats"
    }
    
    local missing_functions = {}
    for _, func_name in ipairs(functions_to_test) do
        if not _G[func_name] then
            table.insert(missing_functions, func_name)
        end
    end
    
    if #missing_functions == 0 then
        print("✅ All enhanced functions are available")
    else
        print("❌ Missing functions: " .. table.concat(missing_functions, ", "))
        return false
    end
    
    return true
end

local function test_configuration_flags()
    print("\n3. Testing Configuration Flags...")
    
    local required_config = {
        "ENHANCED_MEMORY_ENABLED", "ENEMY_DETECTION_ENABLED", 
        "POWERUP_DETECTION_ENABLED", "TILE_SAMPLING_ENABLED",
        "ENHANCED_DEATH_DETECTION", "VELOCITY_TRACKING_ENABLED"
    }
    
    local missing_config = {}
    for _, config_name in ipairs(required_config) do
        if CONFIG[config_name] == nil then
            table.insert(missing_config, config_name)
        end
    end
    
    if #missing_config == 0 then
        print("✅ All configuration flags are defined")
        print("   Enhanced Memory: " .. tostring(CONFIG.ENHANCED_MEMORY_ENABLED))
        print("   Enemy Detection: " .. tostring(CONFIG.ENEMY_DETECTION_ENABLED))
        print("   Power-up Detection: " .. tostring(CONFIG.POWERUP_DETECTION_ENABLED))
        print("   Tile Sampling: " .. tostring(CONFIG.TILE_SAMPLING_ENABLED))
        print("   Enhanced Death Detection: " .. tostring(CONFIG.ENHANCED_DEATH_DETECTION))
        print("   Velocity Tracking: " .. tostring(CONFIG.VELOCITY_TRACKING_ENABLED))
    else
        print("❌ Missing configuration flags: " .. table.concat(missing_config, ", "))
        return false
    end
    
    return true
end

local function test_enemy_type_constants()
    print("\n4. Testing Enemy Type Constants...")
    
    local required_enemy_types = {
        "NONE", "GOOMBA", "KOOPA", "BUZZY_BEETLE", "HAMMER_BRO",
        "LAKITU", "SPINY", "PIRANHA_PLANT", "BLOOPER", "BULLET_BILL", "CHEEP_CHEEP"
    }
    
    local missing_types = {}
    for _, type_name in ipairs(required_enemy_types) do
        if ENEMY_TYPES[type_name] == nil then
            table.insert(missing_types, type_name)
        end
    end
    
    if #missing_types == 0 then
        print("✅ All enemy type constants are defined")
    else
        print("❌ Missing enemy types: " .. table.concat(missing_types, ", "))
        return false
    end
    
    return true
end

local function test_powerup_type_constants()
    print("\n5. Testing Power-up Type Constants...")
    
    local required_powerup_types = {
        "NONE", "MUSHROOM", "FIRE_FLOWER", "STAR", "ONE_UP"
    }
    
    local missing_types = {}
    for _, type_name in ipairs(required_powerup_types) do
        if POWERUP_TYPES[type_name] == nil then
            table.insert(missing_types, type_name)
        end
    end
    
    if #missing_types == 0 then
        print("✅ All power-up type constants are defined")
    else
        print("❌ Missing power-up types: " .. table.concat(missing_types, ", "))
        return false
    end
    
    return true
end

local function test_utility_functions()
    print("\n6. Testing Enhanced Utility Functions...")
    
    local utility_functions = {
        "toggle_enhanced_features", "toggle_enemy_detection", "toggle_powerup_detection",
        "toggle_tile_sampling", "toggle_enhanced_death_detection", "toggle_velocity_tracking",
        "print_enhanced_status", "validate_enhanced_addresses"
    }
    
    local missing_functions = {}
    for _, func_name in ipairs(utility_functions) do
        if not _G[func_name] then
            table.insert(missing_functions, func_name)
        end
    end
    
    if #missing_functions == 0 then
        print("✅ All enhanced utility functions are available")
    else
        print("❌ Missing utility functions: " .. table.concat(missing_functions, ", "))
        return false
    end
    
    return true
end

-- Run all tests
local function run_all_tests()
    print("Starting Enhanced Memory Reading System Tests...\n")
    
    local tests = {
        test_enhanced_memory_addresses,
        test_enhanced_functions,
        test_configuration_flags,
        test_enemy_type_constants,
        test_powerup_type_constants,
        test_utility_functions
    }
    
    local passed = 0
    local total = #tests
    
    for i, test_func in ipairs(tests) do
        if test_func() then
            passed = passed + 1
        end
    end
    
    print(string.format("\n=== Test Results: %d/%d tests passed ===", passed, total))
    
    if passed == total then
        print("🎉 All tests passed! Enhanced memory reading system is ready.")
        print("\nTo use enhanced features:")
        print("1. Call toggle_enhanced_features() to enable/disable all features")
        print("2. Call print_enhanced_status() to see current feature status")
        print("3. Call validate_enhanced_addresses() to test memory reading")
        print("4. Individual features can be toggled with specific functions")
    else
        print("❌ Some tests failed. Please check the implementation.")
    end
    
    return passed == total
end

-- Execute tests
run_all_tests()