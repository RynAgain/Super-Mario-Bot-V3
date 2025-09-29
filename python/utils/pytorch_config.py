"""
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
