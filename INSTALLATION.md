# Installation Guide - Super Mario Bros AI Training System

This guide provides comprehensive installation instructions for the Super Mario Bros AI Training System on Windows, macOS, and Linux.

## 📋 System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 8GB system memory
- **Storage**: 2GB free space
- **Internet**: Required for dependency installation

### Recommended Requirements
- **Operating System**: Windows 11, macOS 12+, or Ubuntu 20.04+
- **Python**: 3.9 or 3.10
- **GPU**: NVIDIA GPU with 4GB+ VRAM and CUDA support
- **RAM**: 16GB system memory
- **Storage**: 5GB free space (for logs and checkpoints)

### Required Software
- **FCEUX Emulator**: Version 2.6.4 or higher
- **Super Mario Bros ROM**: Legally obtained NES ROM file
- **Git**: For cloning the repository

## 🚀 Quick Installation (Windows)

The easiest way to install on Windows is using our automated installation script:

### Step 1: Clone Repository
```batch
git clone https://github.com/RynAgain/Super-Mario-Bot-V3.git
cd Super-Mario-Bot-V3
```

### Step 2: Run Automated Installer
```batch
install.bat
```

The installer will:
- Check Python version compatibility
- Create a virtual environment (optional)
- Install all Python dependencies
- Install the Super Mario AI package
- Run system validation tests
- Provide next steps guidance

### Step 3: Start Training
```batch
run_training.bat
```

## 🔧 Manual Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/RynAgain/Super-Mario-Bot-V3.git
cd Super-Mario-Bot-V3
```

### Step 2: Python Environment Setup

#### Option A: Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv mario_ai_env

# Activate virtual environment
# Windows:
mario_ai_env\Scripts\activate
# macOS/Linux:
source mario_ai_env/bin/activate
```

#### Option B: Conda Environment
```bash
# Create conda environment
conda create -n mario_ai python=3.9
conda activate mario_ai
```

### Step 3: Install Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Step 4: Verify Installation
```bash
# Run system validation
python validate_system.py

# Run comprehensive tests
python test_complete_system_integration.py
```

## 🎮 FCEUX Emulator Setup

### Download and Install FCEUX

#### Windows
1. Download FCEUX from http://fceux.com/web/download.html
2. Extract to a folder (e.g., `C:\FCEUX\`)
3. Add FCEUX to your PATH (optional)

#### macOS
```bash
# Using Homebrew
brew install fceux

# Or download from website
# http://fceux.com/web/download.html
```

#### Linux (Ubuntu/Debian)
```bash
# Install from package manager
sudo apt update
sudo apt install fceux

# Or compile from source
git clone https://github.com/TASVideos/fceux.git
cd fceux
mkdir build && cd build
cmake ..
make
sudo make install
```

### Configure FCEUX

1. **Launch FCEUX**
2. **Load ROM**: `File > Open ROM` → Select your Super Mario Bros ROM
3. **Load Lua Script**: `File > Lua > New Lua Script Window`
4. **Browse to Script**: Navigate to `lua/mario_ai.lua`
5. **Run Script**: Click "Run" - should show "Waiting for connection..."

## 🐍 Python Dependencies Explained

### Core Dependencies
- **torch**: PyTorch deep learning framework
- **torchvision**: Computer vision utilities
- **numpy**: Numerical computing
- **opencv-python**: Image processing
- **websockets**: WebSocket communication
- **pyyaml**: YAML configuration parsing
- **psutil**: System monitoring

### Optional Dependencies
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical plotting
- **tensorboard**: Training visualization
- **pytest**: Testing framework

### GPU Support (Optional)
For NVIDIA GPU acceleration:
```bash
# Uninstall CPU-only PyTorch
pip uninstall torch torchvision

# Install GPU version (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🔍 Installation Verification

### Quick System Check
```bash
# Check Python version
python --version

# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check GPU availability (if applicable)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check package installation
python -c "import python.main; print('Super Mario AI package installed successfully')"
```

### Comprehensive Validation
```bash
# Run full system validation
python validate_system.py

# Expected output:
# ✓ Python version check passed
# ✓ Dependencies check passed
# ✓ Configuration files valid
# ✓ Neural network components working
# ✓ Communication system ready
# ✓ System validation complete
```

### Integration Tests
```bash
# Run comprehensive integration tests
python test_complete_system_integration.py

# This will test:
# - End-to-end workflow simulation
# - Error handling scenarios
# - Performance and synchronization
# - CSV logging system
# - Checkpoint system
```

## 🛠️ Troubleshooting Installation Issues

### Python Version Issues
```bash
# Error: Python version too old
# Solution: Install Python 3.8+ from python.org

# Check available Python versions
python3 --version
python3.9 --version
python3.10 --version

# Use specific version
python3.9 -m venv mario_ai_env
```

### Dependency Installation Failures

#### PyTorch Installation Issues
```bash
# Clear pip cache
pip cache purge

# Install with no cache
pip install --no-cache-dir torch torchvision

# For older systems, use CPU-only version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### OpenCV Installation Issues
```bash
# Alternative OpenCV installation
pip uninstall opencv-python
pip install opencv-python-headless

# Or use conda
conda install opencv
```

#### WebSocket Issues
```bash
# Alternative websockets installation
pip uninstall websockets
pip install websockets==10.4
```

### Permission Issues

#### Windows
```batch
# Run as administrator if needed
# Or install to user directory
pip install --user -r requirements.txt
```

#### macOS/Linux
```bash
# Use user installation
pip install --user -r requirements.txt

# Or fix permissions
sudo chown -R $USER ~/.local/
```

### Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf mario_ai_env
python -m venv mario_ai_env

# Ensure activation works
source mario_ai_env/bin/activate  # macOS/Linux
mario_ai_env\Scripts\activate     # Windows
```

### FCEUX Issues

#### FCEUX Not Found
- Ensure FCEUX is installed and in PATH
- Try absolute path to FCEUX executable
- Check FCEUX version (2.6.4+ required)

#### Lua Script Issues
- Verify `lua/mario_ai.lua` exists
- Check FCEUX Lua console for errors
- Ensure ROM is loaded before running script

#### Connection Issues
- Check Windows Firewall settings
- Verify port 8765 is available
- Try different port in configuration

## 🔧 Advanced Installation Options

### Development Installation
```bash
# Clone with development tools
git clone --recurse-submodules https://github.com/your-username/Super-Mario-Bot-V3.git

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Docker Installation (Experimental)
```bash
# Build Docker image
docker build -t super-mario-ai .

# Run container
docker run -it --gpus all -p 8765:8765 super-mario-ai
```

### Custom Configuration
```bash
# Copy default configurations
cp config/training_config.yaml config/my_training_config.yaml

# Edit configuration
nano config/my_training_config.yaml

# Use custom configuration
python python/main.py train --config config/my_training_config.yaml
```

## 📁 Directory Structure After Installation

```
Super-Mario-Bot-V3/
├── mario_ai_env/           # Virtual environment (if created)
├── logs/                   # Training logs (created during training)
├── checkpoints/            # Model checkpoints (created during training)
├── plots/                  # Performance plots (created during analysis)
├── config/                 # Configuration files
├── docs/                   # Documentation
├── examples/               # Example configurations and scripts
├── lua/                    # FCEUX Lua scripts
├── python/                 # Python source code
├── test_*.py              # Test files
├── setup.py               # Package setup
├── install.bat            # Windows installer
├── run_training.bat       # Windows training launcher
├── validate_system.py     # System validation
└── requirements.txt       # Python dependencies
```

## 🎯 Next Steps

After successful installation:

1. **Validate System**: Run `python validate_system.py`
2. **Run Tests**: Execute `python test_complete_system_integration.py`
3. **Setup FCEUX**: Install emulator and load ROM
4. **Start Training**: Use `run_training.bat` or `python python/main.py train`
5. **Monitor Progress**: Check `logs/` directory for training data
6. **Generate Plots**: Use `python python/logging/plotter.py`

## 📞 Getting Help

If you encounter issues during installation:

1. **Check System Requirements**: Ensure your system meets minimum requirements
2. **Review Error Messages**: Look for specific error details
3. **Check Troubleshooting Section**: See common solutions above
4. **Run Validation**: Use `python validate_system.py` to identify issues
5. **Create Issue**: Report problems on GitHub with system details

For additional support, see:
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
- [USAGE.md](USAGE.md) - Usage examples and guides
- [GitHub Issues](https://github.com/your-username/Super-Mario-Bot-V3/issues) - Community support

---

**Happy Training!** 🎮🚀