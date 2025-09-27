"""
Preprocessing Utilities for Super Mario Bros AI Training

This module implements preprocessing functions for frame stacking, image processing,
state normalization, and data type conversions for PyTorch.

Features:
- Frame stacking (8 frames as specified)
- Image preprocessing (resize, grayscale, normalization)
- State normalization for memory features
- Data type conversions for PyTorch
- Efficient memory management
"""

import torch
import numpy as np
import cv2
from collections import deque
from typing import Tuple, Optional, Union, List, Dict, Any
import logging


class FrameStack:
    """
    Manages frame stacking for temporal information in DQN training.
    
    Maintains a circular buffer of preprocessed frames and provides
    efficient stacking operations for neural network input.
    """
    
    def __init__(
        self,
        stack_size: int = 8,
        frame_size: Tuple[int, int] = (84, 84),
        device: str = "cpu"
    ):
        """
        Initialize frame stack manager.
        
        Args:
            stack_size: Number of frames to stack (8 for Mario)
            frame_size: Target frame dimensions (84, 84)
            device: Device for tensor operations
        """
        self.stack_size = stack_size
        self.frame_size = frame_size
        self.device = torch.device(device)
        
        # Initialize frame buffer
        self.frames = deque(maxlen=stack_size)
        self.state_vectors = deque(maxlen=stack_size)
        
        # Pre-allocate zero frame for initialization
        self.zero_frame = torch.zeros(frame_size, dtype=torch.float32, device=self.device)
        self.zero_state = torch.zeros(12, dtype=torch.float32, device=self.device)
        
        # Initialize with zero frames
        self.reset()
    
    def reset(self):
        """Reset frame stack with zero frames."""
        self.frames.clear()
        self.state_vectors.clear()
        
        for _ in range(self.stack_size):
            self.frames.append(self.zero_frame.clone())
            self.state_vectors.append(self.zero_state.clone())
    
    def add_frame(
        self,
        frame: Union[torch.Tensor, np.ndarray],
        state_vector: Union[torch.Tensor, np.ndarray]
    ):
        """
        Add a new frame and state vector to the stack.
        
        Args:
            frame: Preprocessed frame (84, 84)
            state_vector: Game state vector (12,)
        """
        # Convert to tensors if necessary
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame).float().to(self.device)
        if isinstance(state_vector, np.ndarray):
            state_vector = torch.from_numpy(state_vector).float().to(self.device)
        
        # Ensure correct device
        frame = frame.to(self.device)
        state_vector = state_vector.to(self.device)
        
        # Add to buffers
        self.frames.append(frame)
        self.state_vectors.append(state_vector)
    
    def get_stacked_input(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current stacked frames and most recent state vector.
        
        Returns:
            Tuple of (stacked_frames, state_vector)
            - stacked_frames: (stack_size, height, width)
            - state_vector: (12,)
        """
        # Stack frames along first dimension
        stacked_frames = torch.stack(list(self.frames), dim=0)
        
        # Use most recent state vector
        current_state = self.state_vectors[-1] if self.state_vectors else self.zero_state
        
        return stacked_frames, current_state
    
    def get_batch_input(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get stacked input with batch dimension.
        
        Returns:
            Tuple of (stacked_frames, state_vector) with batch dimension
            - stacked_frames: (1, stack_size, height, width)
            - state_vector: (1, 12)
        """
        stacked_frames, state_vector = self.get_stacked_input()
        return stacked_frames.unsqueeze(0), state_vector.unsqueeze(0)


class FramePreprocessor:
    """
    Handles frame preprocessing operations for consistent input format.
    
    Converts raw game frames to the format expected by the neural network.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (84, 84),
        grayscale: bool = True,
        normalize: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize frame preprocessor.
        
        Args:
            target_size: Target frame dimensions
            grayscale: Convert to grayscale
            normalize: Normalize pixel values to [0, 1]
            device: Device for tensor operations
        """
        self.target_size = target_size
        self.grayscale = grayscale
        self.normalize = normalize
        self.device = torch.device(device)
    
    def preprocess_frame(
        self,
        frame: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Preprocess a single frame.
        
        Args:
            frame: Raw frame (H, W, C) or (H, W)
            
        Returns:
            Preprocessed frame tensor (84, 84)
        """
        # Convert to numpy if tensor
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        
        # Ensure uint8 format for OpenCV
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        
        # Convert to grayscale if needed
        if self.grayscale and len(frame.shape) == 3:
            if frame.shape[2] == 3:  # RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            elif frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        
        # Resize to target size
        if frame.shape[:2] != self.target_size:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame).float().to(self.device)
        
        # Normalize to [0, 1] if requested
        if self.normalize:
            frame_tensor = frame_tensor / 255.0
        
        return frame_tensor
    
    def preprocess_batch(
        self,
        frames: List[Union[torch.Tensor, np.ndarray]]
    ) -> torch.Tensor:
        """
        Preprocess a batch of frames.
        
        Args:
            frames: List of raw frames
            
        Returns:
            Batch of preprocessed frames (batch_size, 84, 84)
        """
        processed_frames = []
        for frame in frames:
            processed_frame = self.preprocess_frame(frame)
            processed_frames.append(processed_frame)
        
        return torch.stack(processed_frames, dim=0)


class StateNormalizer:
    """
    Normalizes game state vectors for consistent neural network input.
    
    Handles normalization of various game state features including
    positions, velocities, and categorical variables.
    """
    
    def __init__(self):
        """Initialize state normalizer with Mario-specific parameters."""
        # Normalization parameters for Mario game state
        self.normalization_params = {
            'mario_x_max': 3168.0,      # Level 1-1 length
            'mario_y_max': 240.0,       # Screen height
            'velocity_max': 127.0,      # Maximum velocity value
            'lives_max': 5.0,           # Assumed maximum lives
            'timer_max': 400.0,         # Maximum timer value
            'score_max': 999999.0,      # Maximum score
            'coins_max': 99.0           # Maximum coins
        }
    
    def normalize_state_vector(
        self,
        game_state: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Normalize game state dictionary to feature vector.
        
        Args:
            game_state: Dictionary containing game state values
            
        Returns:
            Normalized state vector tensor (12,)
        """
        # Extract and normalize features
        features = []
        
        # Mario position (normalized 0-1)
        mario_x_norm = game_state.get('mario_x', 0) / self.normalization_params['mario_x_max']
        mario_y_norm = game_state.get('mario_y', 0) / self.normalization_params['mario_y_max']
        features.extend([mario_x_norm, mario_y_norm])
        
        # Mario velocity (normalized -1 to 1)
        mario_x_vel_norm = game_state.get('mario_x_vel', 0) / self.normalization_params['velocity_max']
        mario_y_vel_norm = game_state.get('mario_y_vel', 0) / self.normalization_params['velocity_max']
        features.extend([mario_x_vel_norm, mario_y_vel_norm])
        
        # Power state (one-hot encoding)
        power_state = game_state.get('power_state', 0)
        power_state_small = 1.0 if power_state == 0 else 0.0
        power_state_big = 1.0 if power_state == 1 else 0.0
        power_state_fire = 1.0 if power_state == 2 else 0.0
        features.extend([power_state_small, power_state_big, power_state_fire])
        
        # Boolean flags
        on_ground = float(game_state.get('on_ground', 0))
        direction = float(game_state.get('direction', 0))  # 0 or 1
        features.extend([on_ground, direction])
        
        # Lives (normalized 0-1)
        lives_norm = game_state.get('lives', 3) / self.normalization_params['lives_max']
        features.append(lives_norm)
        
        # Invincibility flag
        invincible = float(game_state.get('invincible', 0) > 0)
        features.append(invincible)
        
        # Level progress (0-1)
        level_progress = mario_x_norm  # Same as normalized X position
        features.append(level_progress)
        
        # Ensure we have exactly 12 features
        assert len(features) == 12, f"Expected 12 features, got {len(features)}"
        
        return torch.tensor(features, dtype=torch.float32)
    
    def normalize_batch_states(
        self,
        game_states: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """
        Normalize a batch of game states.
        
        Args:
            game_states: List of game state dictionaries
            
        Returns:
            Batch of normalized state vectors (batch_size, 12)
        """
        normalized_states = []
        for state in game_states:
            normalized_state = self.normalize_state_vector(state)
            normalized_states.append(normalized_state)
        
        return torch.stack(normalized_states, dim=0)


class DataConverter:
    """
    Handles data type conversions and device management for PyTorch.
    
    Provides utilities for converting between numpy arrays and PyTorch tensors,
    managing device placement, and handling batch operations.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize data converter.
        
        Args:
            device: Target device for tensors
        """
        self.device = torch.device(device)
    
    def to_tensor(
        self,
        data: Union[np.ndarray, List, float, int],
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Convert data to PyTorch tensor.
        
        Args:
            data: Input data
            dtype: Target tensor dtype
            
        Returns:
            PyTorch tensor on specified device
        """
        if isinstance(data, torch.Tensor):
            return data.to(dtype).to(self.device)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(dtype).to(self.device)
        else:
            return torch.tensor(data, dtype=dtype, device=self.device)
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert PyTorch tensor to numpy array.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Numpy array
        """
        return tensor.detach().cpu().numpy()
    
    def prepare_batch(
        self,
        frames: List[torch.Tensor],
        state_vectors: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch tensors for neural network input.
        
        Args:
            frames: List of frame stacks
            state_vectors: List of state vectors
            
        Returns:
            Tuple of batched tensors
        """
        # Stack into batches
        batch_frames = torch.stack(frames, dim=0).to(self.device)
        batch_states = torch.stack(state_vectors, dim=0).to(self.device)
        
        return batch_frames, batch_states


class MarioPreprocessor:
    """
    Complete preprocessing pipeline for Super Mario Bros AI.
    
    Combines frame preprocessing, state normalization, and frame stacking
    into a single convenient interface.
    """
    
    def __init__(
        self,
        stack_size: int = 8,
        frame_size: Tuple[int, int] = (84, 84),
        device: str = "cpu"
    ):
        """
        Initialize Mario preprocessor.
        
        Args:
            stack_size: Number of frames to stack
            frame_size: Target frame dimensions
            device: Device for tensor operations
        """
        self.frame_stack = FrameStack(stack_size, frame_size, device)
        self.frame_preprocessor = FramePreprocessor(frame_size, device=device)
        self.state_normalizer = StateNormalizer()
        self.data_converter = DataConverter(device)
        
        self.device = torch.device(device)
    
    def reset(self):
        """Reset the preprocessor state."""
        self.frame_stack.reset()
    
    def process_step(
        self,
        raw_frame: Union[torch.Tensor, np.ndarray],
        game_state: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single step (frame + game state).
        
        Args:
            raw_frame: Raw game frame
            game_state: Game state dictionary
            
        Returns:
            Tuple of (stacked_frames, state_vector) ready for network input
        """
        # Preprocess frame
        processed_frame = self.frame_preprocessor.preprocess_frame(raw_frame)
        
        # Normalize state
        normalized_state = self.state_normalizer.normalize_state_vector(game_state)
        normalized_state = normalized_state.to(self.device)
        
        # Add to frame stack
        self.frame_stack.add_frame(processed_frame, normalized_state)
        
        # Get stacked input
        return self.frame_stack.get_batch_input()
    
    def get_current_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current stacked state without adding new frame.
        
        Returns:
            Current stacked frames and state vector with batch dimension
        """
        return self.frame_stack.get_batch_input()


if __name__ == "__main__":
    # Test preprocessing components
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing preprocessing on device: {device}")
    
    # Test frame preprocessor
    print("\nTesting frame preprocessor...")
    frame_preprocessor = FramePreprocessor(device=device)
    
    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8)
    processed_frame = frame_preprocessor.preprocess_frame(dummy_frame)
    print(f"Processed frame shape: {processed_frame.shape}")
    print(f"Processed frame range: {processed_frame.min():.3f} - {processed_frame.max():.3f}")
    
    # Test state normalizer
    print("\nTesting state normalizer...")
    state_normalizer = StateNormalizer()
    
    dummy_game_state = {
        'mario_x': 1000,
        'mario_y': 120,
        'mario_x_vel': 20,
        'mario_y_vel': -10,
        'power_state': 1,
        'on_ground': 1,
        'direction': 1,
        'lives': 3,
        'invincible': 0
    }
    
    normalized_state = state_normalizer.normalize_state_vector(dummy_game_state)
    print(f"Normalized state shape: {normalized_state.shape}")
    print(f"Normalized state: {normalized_state}")
    
    # Test frame stack
    print("\nTesting frame stack...")
    frame_stack = FrameStack(device=device)
    
    # Add some frames
    for i in range(10):
        frame = torch.randn(84, 84)
        state = torch.randn(12)
        frame_stack.add_frame(frame, state)
    
    stacked_frames, current_state = frame_stack.get_batch_input()
    print(f"Stacked frames shape: {stacked_frames.shape}")
    print(f"Current state shape: {current_state.shape}")
    
    # Test complete preprocessor
    print("\nTesting complete Mario preprocessor...")
    mario_preprocessor = MarioPreprocessor(device=device)
    
    # Process a step
    raw_frame = np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8)
    game_state = dummy_game_state
    
    stacked_frames, state_vector = mario_preprocessor.process_step(raw_frame, game_state)
    print(f"Final output shapes:")
    print(f"Stacked frames: {stacked_frames.shape}")
    print(f"State vector: {state_vector.shape}")
    
    print("\nPreprocessing tests completed successfully!")