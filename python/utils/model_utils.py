"""
Model Utilities for Super Mario Bros AI Training

This module provides utilities for model management including:
- Model saving/loading
- Checkpoint management
- Network parameter initialization
- Device management (CPU/GPU)
- Model optimization and compilation
"""

import torch
import torch.nn as nn
import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
from datetime import datetime
import pickle


class ModelManager:
    """
    Manages model saving, loading, and checkpoint operations.
    
    Provides comprehensive model management including versioning,
    metadata tracking, and automatic cleanup of old checkpoints.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        save_optimizer: bool = True,
        compression: bool = True
    ):
        """
        Initialize model manager.
        
        Args:
            checkpoint_dir: Directory for saving checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_optimizer: Whether to save optimizer state
            compression: Whether to compress checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.compression = compression
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        episode: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Save model checkpoint with metadata.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state to save (optional)
            episode: Current episode number
            step: Current step number
            metrics: Training metrics dictionary
            metadata: Additional metadata
            filename: Custom filename (optional)
            
        Returns:
            Path to saved checkpoint file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_ep{episode:06d}_step{step:08d}_{timestamp}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'episode': episode,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'model_class': model.__class__.__name__,
            'model_config': getattr(model, 'config', {}),
            'metrics': metrics or {},
            'metadata': metadata or {}
        }
        
        # Add optimizer state if requested
        if self.save_optimizer and optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint_data['optimizer_class'] = optimizer.__class__.__name__
        
        # Save checkpoint
        if self.compression:
            torch.save(checkpoint_data, checkpoint_path, _use_new_zipfile_serialization=True)
        else:
            torch.save(checkpoint_data, checkpoint_path)
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            device: Device to load tensors to
            strict: Whether to strictly enforce state dict keys
            
        Returns:
            Checkpoint metadata dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint_data['model_state_dict'], strict=strict)
        
        # Load optimizer state if available
        if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
        return {
            'episode': checkpoint_data.get('episode', 0),
            'step': checkpoint_data.get('step', 0),
            'timestamp': checkpoint_data.get('timestamp', ''),
            'metrics': checkpoint_data.get('metrics', {}),
            'metadata': checkpoint_data.get('metadata', {})
        }
    
    def save_best_model(
        self,
        model: nn.Module,
        metric_value: float,
        metric_name: str = "reward",
        higher_better: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save model if it's the best so far based on a metric.
        
        Args:
            model: Model to potentially save
            metric_value: Current metric value
            metric_name: Name of the metric
            higher_better: Whether higher values are better
            metadata: Additional metadata
            
        Returns:
            True if model was saved as new best
        """
        best_model_path = self.checkpoint_dir / "best_model.pt"
        best_info_path = self.checkpoint_dir / "best_model_info.json"
        
        # Check if this is the best model so far
        is_best = False
        
        if best_info_path.exists():
            with open(best_info_path, 'r') as f:
                best_info = json.load(f)
            
            current_best = best_info.get(metric_name, float('-inf') if higher_better else float('inf'))
            
            if higher_better:
                is_best = metric_value > current_best
            else:
                is_best = metric_value < current_best
        else:
            is_best = True
        
        if is_best:
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }, best_model_path)
            
            # Save info
            best_info = {
                metric_name: metric_value,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            with open(best_info_path, 'w') as f:
                json.dump(best_info, f, indent=2)
            
            self.logger.info(f"Saved new best model with {metric_name}: {metric_value}")
        
        return is_best
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with metadata.
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pt"):
            try:
                # Load minimal metadata
                checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
                
                checkpoints.append({
                    'filename': checkpoint_file.name,
                    'path': str(checkpoint_file),
                    'episode': checkpoint_data.get('episode', 0),
                    'step': checkpoint_data.get('step', 0),
                    'timestamp': checkpoint_data.get('timestamp', ''),
                    'metrics': checkpoint_data.get('metrics', {}),
                    'size_mb': checkpoint_file.stat().st_size / (1024 * 1024)
                })
            except Exception as e:
                self.logger.warning(f"Could not read checkpoint {checkpoint_file}: {e}")
        
        # Sort by episode and step
        checkpoints.sort(key=lambda x: (x['episode'], x['step']))
        
        return checkpoints
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints limit."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) > self.max_checkpoints:
            # Remove oldest checkpoints
            to_remove = checkpoints[:-self.max_checkpoints]
            
            for checkpoint in to_remove:
                try:
                    os.remove(checkpoint['path'])
                    self.logger.info(f"Removed old checkpoint: {checkpoint['filename']}")
                except Exception as e:
                    self.logger.warning(f"Could not remove checkpoint {checkpoint['filename']}: {e}")


class DeviceManager:
    """
    Manages device selection and tensor operations across CPU/GPU.
    
    Provides utilities for automatic device detection, memory management,
    and efficient tensor operations.
    """
    
    def __init__(self, preferred_device: str = "auto"):
        """
        Initialize device manager.
        
        Args:
            preferred_device: Preferred device ("auto", "cpu", "cuda", or specific GPU)
        """
        self.device = self._select_device(preferred_device)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Using device: {self.device}")
        
        if self.device.type == "cuda":
            self._log_gpu_info()
    
    def _select_device(self, preferred: str) -> torch.device:
        """Select the best available device."""
        if preferred == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        elif preferred == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                self.logger.warning("CUDA requested but not available, using CPU")
                return torch.device("cpu")
        else:
            return torch.device(preferred)
    
    def _log_gpu_info(self):
        """Log GPU information."""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / (1024**3)
            
            self.logger.info(f"GPU Info: {gpu_name} ({gpu_memory:.1f}GB)")
            self.logger.info(f"Available GPUs: {gpu_count}")
    
    def to_device(self, tensor_or_model: Union[torch.Tensor, nn.Module]) -> Union[torch.Tensor, nn.Module]:
        """Move tensor or model to managed device."""
        return tensor_or_model.to(self.device)
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'device': str(self.device)
            }
        else:
            return {'device': str(self.device)}
    
    def clear_cache(self):
        """Clear GPU cache if using CUDA."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            self.logger.info("Cleared GPU cache")


class ModelOptimizer:
    """
    Provides model optimization utilities including compilation and mixed precision.
    
    Handles PyTorch 2.0+ optimizations for improved training and inference performance.
    """
    
    def __init__(self, device_manager: DeviceManager):
        """
        Initialize model optimizer.
        
        Args:
            device_manager: Device manager instance
        """
        self.device_manager = device_manager
        self.logger = logging.getLogger(__name__)
    
    def optimize_model(
        self,
        model: nn.Module,
        compile_model: bool = True,
        mixed_precision: bool = True,
        compile_mode: str = "reduce-overhead"
    ) -> nn.Module:
        """
        Apply optimizations to model.
        
        Args:
            model: Model to optimize
            compile_model: Whether to use torch.compile
            mixed_precision: Whether to enable mixed precision
            compile_mode: Compilation mode for torch.compile
            
        Returns:
            Optimized model
        """
        # Move to device
        model = self.device_manager.to_device(model)
        
        # Apply torch.compile if available and requested
        if compile_model and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode=compile_mode)
                self.logger.info(f"Model compiled with mode: {compile_mode}")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
        
        # Setup mixed precision if requested
        if mixed_precision and self.device_manager.device.type == "cuda":
            self.logger.info("Mixed precision training enabled")
        
        return model
    
    def create_scaler(self) -> Optional[torch.cuda.amp.GradScaler]:
        """Create gradient scaler for mixed precision training."""
        if self.device_manager.device.type == "cuda":
            return torch.cuda.amp.GradScaler()
        return None


class WeightInitializer:
    """
    Provides various weight initialization strategies for neural networks.
    
    Implements common initialization methods optimized for different activation functions.
    """
    
    @staticmethod
    def kaiming_normal_init(module: nn.Module, nonlinearity: str = 'relu'):
        """Apply Kaiming normal initialization."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity=nonlinearity)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    @staticmethod
    def xavier_uniform_init(module: nn.Module):
        """Apply Xavier uniform initialization."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    @staticmethod
    def orthogonal_init(module: nn.Module, gain: float = 1.0):
        """Apply orthogonal initialization."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    @classmethod
    def initialize_model(
        self,
        model: nn.Module,
        method: str = "kaiming_normal",
        **kwargs
    ):
        """
        Initialize all model weights using specified method.
        
        Args:
            model: Model to initialize
            method: Initialization method name
            **kwargs: Additional arguments for initialization method
        """
        if method == "kaiming_normal":
            model.apply(lambda m: self.kaiming_normal_init(m, **kwargs))
        elif method == "xavier_uniform":
            model.apply(self.xavier_uniform_init)
        elif method == "orthogonal":
            model.apply(lambda m: self.orthogonal_init(m, **kwargs))
        else:
            raise ValueError(f"Unknown initialization method: {method}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def model_summary(model: nn.Module, input_shapes: List[Tuple[int, ...]] = None) -> str:
    """
    Generate model summary string.
    
    Args:
        model: PyTorch model
        input_shapes: List of input tensor shapes (without batch dimension)
        
    Returns:
        Model summary string
    """
    param_counts = count_parameters(model)
    
    summary = f"Model: {model.__class__.__name__}\n"
    summary += f"Total parameters: {param_counts['total']:,}\n"
    summary += f"Trainable parameters: {param_counts['trainable']:,}\n"
    summary += f"Non-trainable parameters: {param_counts['non_trainable']:,}\n"
    
    if input_shapes:
        summary += f"Input shapes: {input_shapes}\n"
    
    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    summary += f"Model size: {model_size_mb:.2f} MB\n"
    
    return summary


if __name__ == "__main__":
    # Test model utilities
    from python.models.dueling_dqn import create_dueling_dqn
    
    print("Testing model utilities...")
    
    # Test device manager
    device_manager = DeviceManager()
    print(f"Selected device: {device_manager.device}")
    
    # Test model creation and optimization
    model = create_dueling_dqn()
    optimizer = ModelOptimizer(device_manager)
    model = optimizer.optimize_model(model, compile_model=False)  # Disable compile for testing
    
    # Test model summary
    summary = model_summary(model, [(8, 84, 84), (12,)])
    print(f"\nModel Summary:\n{summary}")
    
    # Test model manager
    model_manager = ModelManager(checkpoint_dir="test_checkpoints")
    
    # Save a test checkpoint
    checkpoint_path = model_manager.save_checkpoint(
        model=model,
        episode=100,
        step=5000,
        metrics={'reward': 1500.0, 'loss': 0.05}
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # List checkpoints
    checkpoints = model_manager.list_checkpoints()
    print(f"Available checkpoints: {len(checkpoints)}")
    
    # Test memory info
    memory_info = device_manager.get_memory_info()
    print(f"Memory info: {memory_info}")
    
    # Cleanup test directory
    import shutil
    if os.path.exists("test_checkpoints"):
        shutil.rmtree("test_checkpoints")
    
    print("Model utilities tests completed!")