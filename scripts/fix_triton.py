#!/usr/bin/env python3
"""
Fix Triton Installation for PyTorch Compilation
This script addresses the Triton dependency issue for PyTorch's torch.compile functionality.
"""

import subprocess
import sys
import os
import platform

def run_command(cmd, description):
    """Run a command and handle errors gracefully."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
        else:
            print(f"❌ {description} failed")
            print(f"Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False
    return True

def check_cuda_version():
    """Check CUDA version to install compatible Triton."""
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            if "CUDA Version:" in output:
                cuda_line = [line for line in output.split('\n') if "CUDA Version:" in line][0]
                cuda_version = cuda_line.split("CUDA Version: ")[1].split()[0]
                print(f"Detected CUDA version: {cuda_version}")
                return cuda_version
    except:
        pass
    return None

def fix_triton_installation():
    """Fix Triton installation for PyTorch compilation."""
    print("🔧 Fixing Triton Installation for PyTorch Compilation")
    print("=" * 60)
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python version: {python_version}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    
    # Check CUDA version
    cuda_version = check_cuda_version()
    
    # Uninstall existing triton if present
    run_command("pip uninstall triton -y", "Uninstalling existing Triton")
    
    # Install compatible Triton version
    if platform.system() == "Windows":
        if cuda_version and cuda_version.startswith("12"):
            # For CUDA 12.x on Windows
            triton_cmd = "pip install triton --index-url https://download.pytorch.org/whl/cu121"
        elif cuda_version and cuda_version.startswith("11"):
            # For CUDA 11.x on Windows
            triton_cmd = "pip install triton --index-url https://download.pytorch.org/whl/cu118"
        else:
            # Default Windows installation
            triton_cmd = "pip install triton"
    else:
        # Linux/Mac installation
        triton_cmd = "pip install triton"
    
    if not run_command(triton_cmd, "Installing compatible Triton"):
        print("\n⚠️  Triton installation failed. Trying alternative approach...")
        
        # Try installing from PyPI directly
        if not run_command("pip install triton==2.1.0", "Installing Triton 2.1.0"):
            print("\n⚠️  Direct Triton installation also failed.")
            print("This may be due to compatibility issues with your system.")
            print("The system will work without Triton, but with reduced performance.")
            return False
    
    # Verify installation
    try:
        import triton
        print(f"✅ Triton successfully installed: version {triton.__version__}")
        return True
    except ImportError:
        print("❌ Triton installation verification failed")
        return False

def create_pytorch_config():
    """Create PyTorch configuration to handle Triton issues gracefully."""
    config_content = '''"""
PyTorch Configuration for Super Mario AI Training System
This module configures PyTorch to handle Triton compilation issues gracefully.
"""

import torch
import os
import warnings

def configure_pytorch_compilation():
    """Configure PyTorch compilation with fallback options."""
    
    # Set environment variables for better PyTorch performance
    os.environ.setdefault('TORCH_COMPILE_DEBUG', '0')
    os.environ.setdefault('TORCHDYNAMO_VERBOSE', '0')
    
    # Try to configure Triton if available
    try:
        import triton
        print(f"✅ Triton available: version {triton.__version__}")
        
        # Set optimal Triton configuration
        os.environ.setdefault('TRITON_CACHE_DIR', './triton_cache')
        
        # Enable torch.compile with Triton backend
        torch.set_float32_matmul_precision('high')
        
        return True
        
    except ImportError:
        print("⚠️  Triton not available - using fallback configuration")
        
        # Disable torch.compile to avoid Triton errors
        torch._dynamo.config.suppress_errors = True
        
        # Use standard PyTorch without compilation
        warnings.filterwarnings("ignore", message=".*triton.*")
        warnings.filterwarnings("ignore", message=".*TensorFloat32.*")
        
        return False

def get_compile_mode():
    """Get the appropriate compile mode based on Triton availability."""
    try:
        import triton
        return "reduce-overhead"  # Use compilation with Triton
    except ImportError:
        return None  # Disable compilation without Triton

# Configure PyTorch on import
configure_pytorch_compilation()
'''
    
    config_path = "python/utils/pytorch_config.py"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ Created PyTorch configuration: {config_path}")

def main():
    """Main function to fix Triton installation."""
    print("🚀 Super Mario AI Training System - Triton Fix")
    print("=" * 60)
    
    # Try to fix Triton installation
    triton_success = fix_triton_installation()
    
    # Create PyTorch configuration regardless
    create_pytorch_config()
    
    print("\n" + "=" * 60)
    if triton_success:
        print("✅ Triton installation fixed successfully!")
        print("   PyTorch compilation optimizations will be available.")
    else:
        print("⚠️  Triton installation could not be completed.")
        print("   The system will work with fallback configuration.")
        print("   Performance may be slightly reduced but functionality is preserved.")
    
    print("\nNext steps:")
    print("1. Restart your Python environment")
    print("2. Run the training system again")
    print("3. The Triton errors should be resolved")

if __name__ == "__main__":
    main()