 # Communication Troubleshooting Guide

This guide helps diagnose and fix communication issues between the Python trainer and FCEUX Lua script.

## Quick Diagnosis

### Step 1: Test Basic Communication
1. Run `test_communication.bat` 
2. Start FCEUX with Super Mario Bros ROM
3. Load `lua/mario_ai.lua` in FCEUX
4. Check if connection is established

### Step 2: Check Logs
- **Python logs**: `logs/communication_test.log`
- **Lua logs**: `logs/lua_debug.log` (if created)

## Common Issues and Solutions

### Issue 1: "LuaSocket not available" in Lua script

**Symptoms:**
- Lua script shows "LuaSocket load failed" error
- No connection established

**Solutions:**
1. **Check FCEUX Version**: Ensure you're using FCEUX 2.6.4 or later
2. **Verify LuaSocket Files**: Check that these files exist in FCEUX directory:
   - `socket.lua`
   - `lua5.1.dll` or `lua51.dll`
3. **Test Socket Loading**: In FCEUX Lua console, try:
   ```lua
   socket = require("socket")
   print("Socket loaded:", socket ~= nil)
   ```

### Issue 2: "Failed to connect to WebSocket server"

**Symptoms:**
- Lua script loads but can't connect
- Python server shows "waiting for connection"

**Solutions:**
1. **Check Python Server**: Ensure `test_communication.py` is running first
2. **Firewall**: Temporarily disable Windows Firewall
3. **Port Conflict**: Try different port:
   ```bash
   python test_communication.py --port 8766
   ```
   And update Lua script CONFIG.WEBSOCKET_PORT = 8766

### Issue 3: "WebSocket handshake failed"

**Symptoms:**
- Connection established but handshake fails
- "Expected 101 Switching Protocols" error

**Solutions:**
1. **Check Server Type**: Ensure Python WebSocket server is running (not HTTP server)
2. **Protocol Version**: Verify WebSocket protocol version compatibility
3. **Headers**: Check if additional headers are needed

### Issue 4: "No game state data received"

**Symptoms:**
- Connection works but no training data
- Empty CSV log files

**Solutions:**
1. **ROM Loading**: Ensure Super Mario Bros ROM is loaded in FCEUX
2. **Game State**: Start the game (not just ROM select screen)
3. **Memory Reading**: Verify memory addresses are correct for your ROM version
4. **Frame Processing**: Check if `process_frame()` is being called

### Issue 5: "Training loop not advancing"

**Symptoms:**
- Game state received but no actions sent back
- Mario doesn't move

**Solutions:**
1. **Action Processing**: Check if actions are being converted to button presses
2. **Controller Input**: Verify `joypad.set()` is working
3. **Frame Sync**: Ensure proper frame synchronization

## Advanced Debugging

### Enable Debug Logging
In `lua/mario_ai.lua`, set:
```lua
CONFIG.DEBUG_ENABLED = true
CONFIG.LOG_TO_FILE = true
CONFIG.LOG_FRAME_SYNC = true
```

### Test Individual Components

#### Test 1: Socket Creation
```lua
-- In FCEUX Lua console
socket = require("socket")
tcp = socket.tcp()
print("TCP socket created:", tcp ~= nil)
if tcp then tcp:close() end
```

#### Test 2: Basic Connection
```lua
-- In FCEUX Lua console
socket = require("socket")
tcp = socket.tcp()
result, err = tcp:connect("localhost", 8765)
print("Connection result:", result, err)
if tcp then tcp:close() end
```

#### Test 3: Memory Reading
```lua
-- In FCEUX Lua console
mario_x = memory.readbyte(0x006D)
mario_y = memory.readbyte(0x00CE)
print("Mario position:", mario_x, mario_y)
```

### Python Server Debug Mode
Run with verbose logging:
```bash
python test_communication.py --duration 60 2>&1 | tee logs/debug_output.txt
```

## File Locations

### Important Files:
- **Main Lua Script**: `lua/mario_ai.lua`
- **Test Script**: `test_communication.py`
- **Python Trainer**: `python/training/trainer.py`
- **WebSocket Server**: `python/communication/websocket_server.py`

### Log Files:
- **Python Logs**: `logs/communication_test.log`
- **Lua Logs**: `logs/lua_debug.log`
- **Training Logs**: `logs/training_YYYYMMDD_HHMMSS.csv`

## Network Configuration

### Default Settings:
- **Host**: localhost (127.0.0.1)
- **Port**: 8765
- **Protocol**: WebSocket (ws://)

### Changing Network Settings:
1. **Python**: Modify `config/network_config.yaml`
2. **Lua**: Update `CONFIG.WEBSOCKET_HOST` and `CONFIG.WEBSOCKET_PORT`

## Performance Optimization

### If Communication is Slow:
1. **Reduce Frame Rate**: Lower target FPS in FCEUX
2. **Optimize Data**: Reduce game state data size
3. **Buffer Size**: Adjust WebSocket buffer sizes
4. **Timeout Values**: Increase timeout values for slower systems

### If Memory Usage is High:
1. **Replay Buffer**: Reduce replay buffer size in training config
2. **Logging**: Disable verbose logging in production
3. **Frame Stack**: Reduce frame stack size

## Getting Help

If issues persist:
1. **Check Logs**: Always check both Python and Lua logs
2. **System Info**: Note your FCEUX version, Python version, OS
3. **Minimal Test**: Use `test_communication.py` to isolate issues
4. **ROM Version**: Verify you're using the correct Super Mario Bros ROM

## Success Indicators

### Working System Shows:
- ✅ "LuaSocket loaded successfully" in Lua
- ✅ "WebSocket connection established" in Lua  
- ✅ "Client connected" in Python
- ✅ "Game states received: X" in test output
- ✅ Mario responds to AI actions in FCEUX
- ✅ Training CSV files contain data (not just headers)

### Healthy Communication:
- Game state data flowing at ~60 FPS
- Actions being sent back to Lua
- Episode progression and resets working
- No timeout errors in logs