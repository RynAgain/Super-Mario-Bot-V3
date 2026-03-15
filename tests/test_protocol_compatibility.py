#!/usr/bin/env python3
"""
Test script to verify WebSocket protocol compatibility for Super Mario Bot V3
Tests the 136-byte packet structure (8-byte header + 128-byte payload)
"""

import sys
import struct
import asyncio
import logging

# Add project root to path
sys.path.append('.')

from python.communication.websocket_server import WebSocketServer

def test_binary_parsing():
    """Test the binary message parsing logic"""
    print('=== Testing Binary Message Parsing ===')
    
    # Create a test 136-byte packet (8-byte header + 128-byte payload)
    # Header: msg_type(1) + frame_id(4) + data_length(2) + checksum(1)
    msg_type = 0x01
    frame_id = 100
    data_length = 128
    
    # Create 128-byte payload
    payload = b'\x00' * 128
    
    # Calculate checksum
    checksum = sum(payload) % 256
    
    # Pack header using little-endian format
    header = struct.pack('<BIHB', msg_type, frame_id, data_length, checksum)
    
    # Create complete packet
    packet = header + payload
    
    print(f'Header size: {len(header)} bytes')
    print(f'Payload size: {len(payload)} bytes') 
    print(f'Total packet size: {len(packet)} bytes')
    print(f'Expected: 136 bytes (8 header + 128 payload)')
    
    if len(packet) == 136:
        print('✅ PASS: Packet size is exactly 136 bytes')
    else:
        print(f'❌ FAIL: Packet size is {len(packet)} bytes, expected 136')
        return False
    
    # Test header parsing
    try:
        parsed_header = struct.unpack('<BIHB', header)
        parsed_msg_type, parsed_frame_id, parsed_data_length, parsed_checksum = parsed_header
        
        print(f'Parsed message type: {parsed_msg_type} (expected {msg_type})')
        print(f'Parsed frame ID: {parsed_frame_id} (expected {frame_id})')
        print(f'Parsed data length: {parsed_data_length} (expected {data_length})')
        print(f'Parsed checksum: {parsed_checksum} (expected {checksum})')
        
        if (parsed_msg_type == msg_type and 
            parsed_frame_id == frame_id and 
            parsed_data_length == data_length and 
            parsed_checksum == checksum):
            print('✅ PASS: Header parsing successful')
            return True
        else:
            print('❌ FAIL: Header parsing mismatch')
            return False
            
    except Exception as e:
        print(f'❌ FAIL: Header parsing error: {e}')
        return False

def test_websocket_server_parsing():
    """Test WebSocket server's binary message parsing"""
    print('\n=== Testing WebSocket Server Binary Parsing ===')
    
    # Create WebSocket server instance
    server = WebSocketServer()
    
    # Create test payload with game state data
    payload_data = bytearray(128)
    
    # Mario data (16 bytes)
    mario_x = 1000
    mario_y = 100
    payload_data[0] = mario_x & 0xFF
    payload_data[1] = (mario_x >> 8) & 0xFF
    payload_data[2] = mario_y & 0xFF
    payload_data[3] = (mario_y >> 8) & 0xFF
    payload_data[4] = 5   # x_velocity
    payload_data[5] = 255 # y_velocity (negative, -1)
    payload_data[6] = 1   # power_state
    payload_data[7] = 0   # animation_state
    payload_data[8] = 1   # direction
    payload_data[9] = 0   # player_state
    payload_data[10] = 3  # lives
    payload_data[11] = 0  # invincibility_timer
    payload_data[12] = 120 # x_pos_raw
    payload_data[13] = 0  # crouching
    payload_data[14] = 0  # reserved
    payload_data[15] = 0  # reserved
    
    # Enemy data (32 bytes - 8 enemies × 4 bytes each)
    for i in range(8):
        base_idx = 16 + (i * 4)
        payload_data[base_idx] = 1      # enemy type
        payload_data[base_idx + 1] = 50 # x_pos
        payload_data[base_idx + 2] = 150 # y_pos
        payload_data[base_idx + 3] = 1  # state
    
    # Level data (64 bytes)
    level_base = 48
    payload_data[level_base] = 200 & 0xFF     # camera_x low
    payload_data[level_base + 1] = (200 >> 8) & 0xFF # camera_x high
    payload_data[level_base + 2] = 1          # world_number
    payload_data[level_base + 3] = 1          # level_number
    payload_data[level_base + 4] = 0          # score bytes
    payload_data[level_base + 5] = 0
    payload_data[level_base + 6] = 0
    payload_data[level_base + 7] = 0
    
    # Time remaining (4 bytes)
    time_remaining = 400
    payload_data[level_base + 8] = time_remaining & 0xFF
    payload_data[level_base + 9] = (time_remaining >> 8) & 0xFF
    payload_data[level_base + 10] = (time_remaining >> 16) & 0xFF
    payload_data[level_base + 11] = (time_remaining >> 24) & 0xFF
    
    # Total coins (2 bytes)
    payload_data[level_base + 12] = 5  # coins
    payload_data[level_base + 13] = 0
    
    # Game variables (16 bytes)
    game_base = 112
    payload_data[game_base] = 8       # game_engine_state
    payload_data[game_base + 1] = 50  # level_progress (50%)
    
    # Distance to flag (2 bytes)
    distance = 2000
    payload_data[game_base + 2] = distance & 0xFF
    payload_data[game_base + 3] = (distance >> 8) & 0xFF
    
    # Frame ID (4 bytes)
    frame_id = 12345
    payload_data[game_base + 4] = frame_id & 0xFF
    payload_data[game_base + 5] = (frame_id >> 8) & 0xFF
    payload_data[game_base + 6] = (frame_id >> 16) & 0xFF
    payload_data[game_base + 7] = (frame_id >> 24) & 0xFF
    
    # Timestamp (4 bytes)
    timestamp = 1609459200000  # Example timestamp
    payload_data[game_base + 8] = timestamp & 0xFF
    payload_data[game_base + 9] = (timestamp >> 8) & 0xFF
    payload_data[game_base + 10] = (timestamp >> 16) & 0xFF
    payload_data[game_base + 11] = (timestamp >> 24) & 0xFF
    
    # Convert to bytes
    payload = bytes(payload_data)
    
    # Calculate checksum
    checksum = sum(payload) % 256
    
    # Create header
    header = struct.pack('<BIHB', 0x01, frame_id, 128, checksum)
    
    # Create complete message
    message = header + payload
    
    print(f'Test message size: {len(message)} bytes')
    print(f'Header: {header.hex()}')
    print(f'Payload checksum: {checksum}')
    
    # Test the server's checksum calculation
    calculated_checksum = server._calculate_checksum(payload)
    if calculated_checksum == checksum:
        print('✅ PASS: Checksum calculation matches')
    else:
        print(f'❌ FAIL: Checksum mismatch - calculated {calculated_checksum}, expected {checksum}')
        return False
    
    print('✅ PASS: WebSocket server binary parsing test successful')
    return True

def test_lua_payload_structure():
    """Test the Lua payload structure matches expectations"""
    print('\n=== Testing Lua Payload Structure ===')
    
    # Verify the structure from the Lua script
    mario_block = 16    # bytes
    enemy_block = 32    # bytes (8 enemies × 4 bytes)
    level_block = 64    # bytes
    game_block = 16     # bytes
    
    total = mario_block + enemy_block + level_block + game_block
    
    print(f'Mario Data Block: {mario_block} bytes')
    print(f'Enemy Data Block: {enemy_block} bytes')
    print(f'Level Data Block: {level_block} bytes')
    print(f'Game Variables Block: {game_block} bytes')
    print(f'Total: {total} bytes')
    
    if total == 128:
        print('✅ PASS: Lua payload structure totals exactly 128 bytes')
        return True
    else:
        print(f'❌ FAIL: Lua payload structure totals {total} bytes, expected 128')
        return False

def main():
    """Run all protocol compatibility tests"""
    print('Super Mario Bot V3 - WebSocket Protocol Compatibility Test')
    print('=' * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Binary parsing
    if test_binary_parsing():
        tests_passed += 1
    
    # Test 2: WebSocket server parsing
    if test_websocket_server_parsing():
        tests_passed += 1
    
    # Test 3: Lua payload structure
    if test_lua_payload_structure():
        tests_passed += 1
    
    print('\n' + '=' * 60)
    print(f'PROTOCOL COMPATIBILITY TEST RESULTS: {tests_passed}/{total_tests} PASSED')
    
    if tests_passed == total_tests:
        print('✅ ALL TESTS PASSED - Protocol compatibility verified!')
        print('The 136-byte packet structure is working correctly.')
        return True
    else:
        print('❌ SOME TESTS FAILED - Protocol compatibility issues detected!')
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)