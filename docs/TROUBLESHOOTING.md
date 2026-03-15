# Troubleshooting Guide - Super Mario Bros AI Training System

This guide provides solutions to common issues and problems you may encounter while using the Super Mario Bros AI Training System.

## 🚨 Quick Diagnostics

### System Health Check
```bash
# Run comprehensive system validation
python validate_system.py

# Run integration tests
python test_complete_system_integration.py

# Check Python environment
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Log Analysis
```bash
# Check recent logs for errors
find logs/ -name "*.csv" -mtime -1 -exec tail -n 20 {} \;

# Check debug events
tail -f logs/latest_session/debug_events.csv

# Monitor system performance
python python/training/training_utils.py --monitor
```

## 🔧 Installation Issues

### Python Version Problems

#### Issue: "Python version too old"
```bash
# Error: Python 3.7 or older detected
# Solution: Install Python 3.8+

# Check available Python versions
python3 --version
python3.8 --version
python3.9 --version

# Use specific Python version
python3.9 -m venv mario_ai_env
python3.9 -m pip install -r requirements.txt
```

#### Issue: "python command not found"
```bash
# Windows: Add Python to PATH
# 1. Find Python installation: where python
# 2. Add to PATH in System Environment Variables
# 3. Restart command prompt

# macOS: Install Python via Homebrew
brew install python@3.9

# Linux: Install Python via package manager
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-pip
```

### Dependency Installation Issues

#### Issue: PyTorch Installation Fails
```bash
# Clear pip cache
pip cache purge

# Install CPU-only version first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For GPU support (CUDA 11.8)
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For older CUDA versions
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

#### Issue: OpenCV Installation Problems
```bash
# Error: "Could not build wheels for opencv-python"
# Solution: Install pre-compiled version
pip uninstall opencv-python
pip install opencv-python-headless

# Alternative: Use conda
conda install opencv

# For headless servers
pip install opencv-python-headless==4.8.0.74
```

#### Issue: WebSocket Installation Fails
```bash
# Error: "Failed building wheel for websockets"
# Solution: Install specific version
pip uninstall websockets
pip install websockets==10.4

# Alternative: Use system packages
sudo apt install python3-websockets  # Linux
brew install python-websockets       # macOS
```

### Virtual Environment Issues

#### Issue: Virtual Environment Won't Activate
```bash
# Windows: Execution policy error
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Alternative activation methods
# Windows:
mario_ai_env\Scripts\activate.bat
# or
mario_ai_env\Scripts\Activate.ps1

# macOS/Linux:
source mario_ai_env/bin/activate
```

#### Issue: "No module named 'python'"
```bash
# Ensure package is installed in virtual environment
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall in development mode
pip uninstall super-mario-ai
pip install -e .
```

## 🎮 FCEUX and ROM Issues

### FCEUX Installation Problems

#### Issue: FCEUX Not Found
```bash
# Windows: Download from http://fceux.com
# Extract to C:\FCEUX\ and add to PATH

# macOS: Install via Homebrew
brew install fceux

# Linux: Install from package manager
sudo apt install fceux

# Verify installation
fceux --help
which fceux
```

#### Issue: FCEUX Version Too Old
```bash
# Check FCEUX version
fceux --version

# Required: FCEUX 2.6.4 or higher
# Download latest from: http://fceux.com/web/download.html

# Compile from source (Linux/macOS)
git clone https://github.com/TASVideos/fceux.git
cd fceux
mkdir build && cd build
cmake ..
make -j4
sudo make install
```

### Lua Script Issues

#### Issue: "Lua script failed to load"
```bash
# Check file exists
ls -la lua/mario_ai.lua

# Verify FCEUX Lua support
fceux --help | grep -i lua

# Check Lua console in FCEUX for error messages
# File > Lua > New Lua Script Window > Check output
```

#### Issue: "JSON module not found"
```bash
# Ensure json.lua is in the same directory
ls -la lua/json.lua

# Check Lua script paths in FCEUX
# Verify lua/mario_ai.lua includes correct path to json.lua
```

### ROM Issues

#### Issue: "ROM file not found"
```bash
# Verify ROM file exists and is readable
ls -la "Super Mario Bros (World).nes"

# Check ROM file permissions
chmod 644 "Super Mario Bros (World).nes"

# Verify ROM is correct version (World/USA)
# File size should be approximately 40KB
```

#### Issue: "Invalid ROM format"
```bash
# ROM must be in .nes format
# Convert from other formats using tools like:
# - Lunar IPS (for IPS patches)
# - FCEUX built-in converter

# Verify ROM checksum (optional)
md5sum "Super Mario Bros (World).nes"
# Expected: 811b027eaf99c2def7b933c5208636de
```

## 🌐 Network and Communication Issues

### WebSocket Connection Problems

#### Issue: "Connection refused on port 8765"
```bash
# Check if port is in use
netstat -an | grep 8765
lsof -i :8765  # macOS/Linux

# Try different port
python python/main.py train --config config/training_config.yaml
# Edit config to use different port (e.g., 8766)
```

#### Issue: "WebSocket connection timeout"
```bash
# Check firewall settings
# Windows: Allow Python through Windows Firewall
# macOS: System Preferences > Security & Privacy > Firewall
# Linux: sudo ufw allow 8765

# Test connection manually
python -c "
import asyncio
import websockets
async def test():
    try:
        async with websockets.connect('ws://localhost:8765') as ws:
            print('Connection successful')
    except Exception as e:
        print(f'Connection failed: {e}')
asyncio.run(test())
"
```

#### Issue: "Frame data not received"
```bash
# Check FCEUX Lua console for errors
# Verify game is running (not paused)
# Check memory address readings in Lua script

# Test frame capture manually
python python/capture/frame_capture.py --test

# Monitor WebSocket traffic
python python/communication/websocket_server.py --debug
```

### Synchronization Issues

#### Issue: "Frame desync detected"
```bash
# Check sync quality in logs
tail -f logs/latest_session/sync_quality.csv

# Adjust sync parameters in config
# Increase buffer size or timeout values

# Monitor system performance
python python/training/training_utils.py --monitor
```

#### Issue: "High frame drops"
```bash
# Check system performance
top
htop  # Linux/macOS
taskmgr  # Windows

# Reduce processing load
# Lower frame rate in FCEUX
# Reduce batch size in training config
# Disable real-time plotting
```

## 🧠 Training Issues

### Training Won't Start

#### Issue: "No CUDA device available"
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Install CUDA toolkit
# Download from: https://developer.nvidia.com/cuda-downloads

# Use CPU training instead
python python/main.py train --device cpu
```

#### Issue: "Out of memory error"
```bash
# Reduce batch size
# Edit config/training_config.yaml:
training:
  batch_size: 16  # Reduce from 32
  
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Monitor GPU memory
nvidia-smi -l 1

# Use gradient accumulation
training:
  batch_size: 16
  accumulation_steps: 2  # Effective batch size: 32
```

#### Issue: "Model not learning"
```bash
# Check learning rate
# Too high: Loss explodes or oscillates
# Too low: Very slow learning

# Analyze reward distribution
python examples/analyze_results.py --session your_session --rewards

# Check exploration vs exploitation
# Ensure epsilon is decreasing properly

# Verify reward function
python python/environment/reward_calculator.py --test
```

### Poor Training Performance

#### Issue: "Training very slow"
```bash
# Profile training performance
python python/main.py train --profile

# Check GPU utilization
nvidia-smi

# Optimize configuration
performance:
  mixed_precision: true
  compile_model: true
  pin_memory: true
  num_workers: 4
```

#### Issue: "Agent not progressing in game"
```bash
# Check reward function weights
# Increase distance reward scale
rewards:
  distance_reward_scale: 2.0  # Increase from 1.0

# Analyze action distribution
python examples/analyze_results.py --session your_session --actions

# Check exploration parameters
training:
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.9995
```

#### Issue: "High loss, unstable training"
```bash
# Reduce learning rate
training:
  learning_rate: 0.0001  # Reduce from 0.001

# Increase target network update frequency
training:
  target_update_frequency: 2000  # Increase from 1000

# Use gradient clipping
training:
  gradient_clip_norm: 1.0
```

## 📊 Logging and Monitoring Issues

### CSV Logging Problems

#### Issue: "Log files not created"
```bash
# Check permissions
ls -la logs/
chmod 755 logs/

# Verify logging configuration
python python/utils/config_loader.py --validate config/logging_config.yaml

# Test CSV logger manually
python python/logging/csv_logger.py --test
```

#### Issue: "Corrupted log files"
```bash
# Check disk space
df -h

# Verify file integrity
head -n 5 logs/session_*/training_steps.csv
tail -n 5 logs/session_*/training_steps.csv

# Recover from backup
cp logs/session_*/training_steps.csv.backup logs/session_*/training_steps.csv
```

### Plotting Issues

#### Issue: "Matplotlib not available"
```bash
# Install matplotlib
pip install matplotlib seaborn

# For headless systems
pip install matplotlib
export MPLBACKEND=Agg

# Test plotting
python python/logging/plotter.py --test
```

#### Issue: "Plots not generating"
```bash
# Check plot directory permissions
mkdir -p plots/
chmod 755 plots/

# Generate plots manually
python python/logging/plotter.py --session your_session --output plots/

# Check for errors
python python/logging/plotter.py --session your_session --debug
```

## 🔍 Performance Issues

### System Performance

#### Issue: "High CPU usage"
```bash
# Monitor CPU usage
top -p $(pgrep -f python)

# Reduce processing load
training:
  num_workers: 1  # Reduce from 4
  batch_size: 16  # Reduce batch size

# Use CPU affinity
taskset -c 0-3 python python/main.py train  # Linux
```

#### Issue: "High memory usage"
```bash
# Monitor memory usage
python python/training/training_utils.py --memory-profile

# Reduce memory usage
training:
  replay_buffer_size: 50000  # Reduce from 100000
  batch_size: 16            # Reduce batch size

capture:
  frame_stack_size: 4       # Reduce from 8
```

#### Issue: "Disk space issues"
```bash
# Check disk usage
du -sh logs/ checkpoints/ plots/

# Clean old logs
find logs/ -type f -mtime +30 -delete

# Compress checkpoints
gzip checkpoints/*.pth

# Configure log rotation
logging:
  max_log_files: 10
  max_log_size_mb: 100
```

## 🐛 Debugging Techniques

### Enable Debug Logging
```bash
# Run with debug logging
python python/main.py train --log-level DEBUG

# Enable component-specific debugging
export MARIO_AI_DEBUG=1
python python/main.py train
```

### Memory Profiling
```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler python/main.py train --episodes 10
```

### Performance Profiling
```bash
# Install profiling tools
pip install py-spy

# Profile running training
py-spy top --pid $(pgrep -f "python.*main.py")

# Generate flame graph
py-spy record -o profile.svg --pid $(pgrep -f "python.*main.py")
```

### Network Debugging
```bash
# Monitor WebSocket traffic
python python/communication/websocket_server.py --debug

# Test network connectivity
telnet localhost 8765

# Capture network packets
tcpdump -i lo port 8765  # Linux/macOS
```

## 📞 Getting Additional Help

### Diagnostic Information to Collect

When reporting issues, please include:

```bash
# System information
python --version
pip list | grep -E "(torch|numpy|opencv|websockets)"
uname -a  # Linux/macOS
systeminfo  # Windows

# Configuration
cat config/training_config.yaml

# Recent logs
tail -n 50 logs/latest_session/debug_events.csv

# System validation results
python validate_system.py > system_validation.txt 2>&1
```

### Common Log Patterns

#### Normal Operation
```
INFO: Episode 100 completed successfully
INFO: Checkpoint saved: checkpoints/mario_ai_episode_100.pth
INFO: Sync quality: 98.5%
```

#### Warning Signs
```
WARNING: Frame drop detected, sync quality: 85.2%
WARNING: High memory usage: 7.8GB
WARNING: GPU utilization low: 45%
```

#### Error Indicators
```
ERROR: WebSocket connection lost
ERROR: CUDA out of memory
ERROR: Invalid game state received
```

### Support Resources

1. **System Validation**: Always run `python validate_system.py` first
2. **Integration Tests**: Use `python test_complete_system_integration.py`
3. **Documentation**: Check [docs/](docs/) for detailed technical information
4. **GitHub Issues**: Report bugs with diagnostic information
5. **Community**: Join discussions for community support

### Creating Effective Bug Reports

1. **Describe the Problem**: What were you trying to do?
2. **Steps to Reproduce**: Exact commands and configuration used
3. **Expected vs Actual**: What should happen vs what actually happened
4. **Environment**: System info, Python version, dependencies
5. **Logs**: Relevant log excerpts and error messages
6. **Workarounds**: Any temporary solutions you've found

---

**Remember**: Most issues can be resolved by running system validation and checking the logs. When in doubt, start with the basics! 🔧