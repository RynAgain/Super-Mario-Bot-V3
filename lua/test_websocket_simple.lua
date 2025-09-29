--[[
Simple WebSocket Connection Test for FCEUX
This script tests basic WebSocket connectivity without the full AI system
]]

-- Simple debug function
local function debug_print(message)
    print("[DEBUG] " .. message)
    if emu and emu.print then
        emu.print("[DEBUG] " .. message)
    end
end

debug_print("Starting simple WebSocket test...")

-- Try to load socket
local socket_ok, socket_result = pcall(require, "socket")
if not socket_ok then
    debug_print("ERROR: Cannot load LuaSocket - " .. tostring(socket_result))
    debug_print("This means FCEUX doesn't have LuaSocket support")
    debug_print("You may need to:")
    debug_print("1. Use a different FCEUX version with LuaSocket")
    debug_print("2. Install LuaSocket manually")
    debug_print("3. Use a different connection method")
    return
end

local socket = socket_result
debug_print("LuaSocket loaded successfully!")

-- Test basic TCP connection
local function test_connection()
    debug_print("Testing TCP connection to localhost:8765...")
    
    local tcp_socket = socket.tcp()
    if not tcp_socket then
        debug_print("ERROR: Failed to create TCP socket")
        return false
    end
    
    tcp_socket:settimeout(5)
    
    local result, err = tcp_socket:connect("localhost", 8765)
    if not result then
        debug_print("ERROR: Failed to connect - " .. (err or "unknown error"))
        debug_print("Make sure the Python training system is running!")
        tcp_socket:close()
        return false
    end
    
    debug_print("SUCCESS: TCP connection established!")
    
    -- Send a simple HTTP request to test
    local http_request = "GET / HTTP/1.1\r\nHost: localhost:8765\r\n\r\n"
    local send_result, send_err = tcp_socket:send(http_request)
    if not send_result then
        debug_print("ERROR: Failed to send data - " .. (send_err or "unknown error"))
        tcp_socket:close()
        return false
    end
    
    debug_print("SUCCESS: Data sent!")
    
    -- Try to receive response
    local response, recv_err = tcp_socket:receive("*l")
    if response then
        debug_print("SUCCESS: Received response - " .. response)
    else
        debug_print("WARNING: No response received - " .. (recv_err or "unknown error"))
    end
    
    tcp_socket:close()
    debug_print("Connection test completed")
    return true
end

-- Run the test
debug_print("=== WebSocket Connection Test ===")
if test_connection() then
    debug_print("=== TEST PASSED ===")
    debug_print("LuaSocket is working in FCEUX!")
    debug_print("The issue may be in the WebSocket handshake or protocol implementation")
else
    debug_print("=== TEST FAILED ===")
    debug_print("LuaSocket connection failed")
    debug_print("Check if Python trainer is running and listening on port 8765")
end

-- Manual test function that can be called from FCEUX console
function test_websocket()
    return test_connection()
end

debug_print("Test script loaded. You can also run test_websocket() manually.")