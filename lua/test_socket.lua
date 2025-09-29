--[[
Simple Socket Test Script for FCEUX
Tests if LuaSocket is working properly
]]

print("=== LuaSocket Test Script ===")

-- Test 1: Try to load socket
print("Test 1: Loading socket library...")
local socket_ok, socket_result = pcall(require, "socket")

if socket_ok then
    print("✅ socket library loaded successfully")
    local socket = socket_result
    
    -- Test 2: Try to load socket.core
    print("Test 2: Loading socket.core...")
    local core_ok, core_result = pcall(require, "socket.core")
    
    if core_ok then
        print("✅ socket.core loaded successfully")
    else
        print("❌ socket.core failed: " .. tostring(core_result))
    end
    
    -- Test 3: Try to create TCP socket
    print("Test 3: Creating TCP socket...")
    local tcp_ok, tcp_socket = pcall(socket.tcp)
    
    if tcp_ok and tcp_socket then
        print("✅ TCP socket created successfully")
        
        -- Test 4: Try to connect to a known server (Google DNS)
        print("Test 4: Testing connection to google.com:80...")
        tcp_socket:settimeout(5)
        local connect_ok, connect_err = tcp_socket:connect("google.com", 80)
        
        if connect_ok then
            print("✅ Connection to google.com successful")
            tcp_socket:close()
        else
            print("❌ Connection failed: " .. tostring(connect_err))
        end
        
        -- Test 5: Try to connect to localhost (our Python server)
        print("Test 5: Testing connection to localhost:8765...")
        local local_socket = socket.tcp()
        local_socket:settimeout(2)
        local local_ok, local_err = local_socket:connect("localhost", 8765)
        
        if local_ok then
            print("✅ Connection to localhost:8765 successful")
            print("   Python WebSocket server is running!")
            local_socket:close()
        else
            print("❌ Connection to localhost:8765 failed: " .. tostring(local_err))
            print("   Make sure Python training system is running first")
        end
        
    else
        print("❌ Failed to create TCP socket: " .. tostring(tcp_socket))
    end
    
else
    print("❌ socket library failed to load: " .. tostring(socket_result))
end

print("\n=== Test Complete ===")
print("If all tests pass, LuaSocket is working correctly")
print("If any tests fail, that indicates the problem area")