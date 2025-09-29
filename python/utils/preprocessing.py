"""
Preprocessing Utilities for Super Mario Bros AI Training

This module implements preprocessing functions for frame stacking, image processing,
state normalization, and data type conversions for PyTorch.

Features:
- Frame stacking (4 frames as specified)
- Image preprocessing (resize, grayscale, normalization)
- State normalization for memory features (12 or 20 features)
- Enhanced binary payload parsing for Lua memory data
- Data type conversions for PyTorch
- Efficient memory management
- Backward compatibility with existing checkpoints
"""

import torch
import numpy as np
import cv2
from collections import deque
from typing import Tuple, Optional, Union, List, Dict, Any
import logging
import struct


class FrameStack:
    """
    Manages frame stacking for temporal information in DQN training.
    
    Maintains a circular buffer of preprocessed frames and provides
    efficient stacking operations for neural network input.
    """
    
    def __init__(
        self,
        stack_size: int = 4,
        frame_size: Tuple[int, int] = (84, 84),
        state_vector_size: int = 12,
        device: str = "cpu"
    ):
        """
        Initialize frame stack manager.
        
        Args:
            stack_size: Number of frames to stack (4 for Mario)
            frame_size: Target frame dimensions (84, 84)
            state_vector_size: Size of state vector (12 or 20)
            device: Device for tensor operations
        """
        self.stack_size = stack_size
        self.frame_size = frame_size
        self.state_vector_size = state_vector_size
        self.device = torch.device(device)
        
        # Initialize frame buffer
        self.frames = deque(maxlen=stack_size)
        self.state_vectors = deque(maxlen=stack_size)
        
        # Pre-allocate zero frame for initialization
        self.zero_frame = torch.zeros(frame_size, dtype=torch.float32, device=self.device)
        self.zero_state = torch.zeros(state_vector_size, dtype=torch.float32, device=self.device)
        
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
            - state_vector: (state_vector_size,)
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
            - state_vector: (1, state_vector_size)
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
    
    def __init__(self, enhanced_features: bool = False):
        """Initialize state normalizer with Mario-specific parameters.
        
        Args:
            enhanced_features: Whether to use enhanced 20-feature mode
        """
        self.enhanced_features = enhanced_features
        self.state_vector_size = 20 if enhanced_features else 12
        
        # Normalization parameters for Mario game state
        self.normalization_params = {
            'mario_x_max': 3168.0,      # Level 1-1 length
            'mario_y_max': 240.0,       # Screen height
            'velocity_max': 127.0,      # Maximum velocity value
            'lives_max': 5.0,           # Assumed maximum lives
            'timer_max': 400.0,         # Maximum timer value
            'score_max': 999999.0,      # Maximum score
            'coins_max': 99.0,          # Maximum coins
            # Enhanced feature parameters
            'enemy_distance_max': 500.0,    # Maximum enemy distance
            'powerup_distance_max': 300.0,  # Maximum power-up distance
            'tile_value_max': 255.0,        # Maximum tile value
            'velocity_magnitude_max': 180.0  # Maximum velocity magnitude (sqrt(127^2 + 127^2))
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
            Normalized state vector tensor (12 or 20 features)
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
        
        # Enhanced features (only if enabled)
        if self.enhanced_features:
            # Enemy threat assessment (2 features)
            closest_enemy_distance = game_state.get('closest_enemy_distance', 999.0)
            enemy_count = game_state.get('enemy_count', 0)
            
            closest_enemy_norm = min(closest_enemy_distance / self.normalization_params['enemy_distance_max'], 1.0)
            enemy_count_norm = min(enemy_count / 5.0, 1.0)  # Max 5 enemies
            features.extend([closest_enemy_norm, enemy_count_norm])
            
            # Power-up detection (2 features)
            powerup_present = float(game_state.get('powerup_present', False))
            powerup_distance = game_state.get('powerup_distance', 999.0)
            powerup_distance_norm = min(powerup_distance / self.normalization_params['powerup_distance_max'], 1.0)
            features.extend([powerup_present, powerup_distance_norm])
            
            # Environmental awareness (2 features)
            solid_tiles_ahead = game_state.get('solid_tiles_ahead', 0)
            pit_detected = float(game_state.get('pit_detected', False))
            solid_tiles_norm = min(solid_tiles_ahead / 10.0, 1.0)  # Max 10 tiles ahead
            features.extend([solid_tiles_norm, pit_detected])
            
            # Enhanced Mario state (2 features)
            velocity_magnitude = game_state.get('velocity_magnitude', 0.0)
            facing_direction = float(game_state.get('facing_direction', 1))  # 0 or 1
            velocity_mag_norm = min(velocity_magnitude / self.normalization_params['velocity_magnitude_max'], 1.0)
            features.extend([velocity_mag_norm, facing_direction])
        
        # Ensure we have the expected number of features
        expected_features = self.state_vector_size
        assert len(features) == expected_features, f"Expected {expected_features} features, got {len(features)}"
        
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


class BinaryPayloadParser:
    """
    Parses binary payload from Lua script containing enhanced memory data.
    
    The Lua script sends a 128-byte binary payload with the following structure:
    - Mario Data Block (16 bytes): positions 1-16
    - Enemy Data Block (32 bytes): positions 17-48
    - Level Data Block (64 bytes): positions 49-112
    - Game Variables Block (16 bytes): positions 113-128
    """
    
    def __init__(self, enhanced_features: bool = False):
        """
        Initialize binary payload parser.
        
        Args:
            enhanced_features: Whether to parse enhanced features from payload
        """
        self.enhanced_features = enhanced_features
    
    def parse_payload(self, payload: bytes) -> Dict[str, Any]:
        """
        Parse 128-byte binary payload from Lua script.
        
        Args:
            payload: 128-byte binary payload
            
        Returns:
            Dictionary containing parsed game state data
        """
        if len(payload) != 128:
            raise ValueError(f"Expected 128-byte payload, got {len(payload)} bytes")
        
        game_state = {}
        
        # Mario Data Block (16 bytes: positions 0-15)
        mario_x_world = struct.unpack('<H', payload[0:2])[0]  # Little-endian uint16
        mario_y_level = struct.unpack('<H', payload[2:4])[0]  # Little-endian uint16
        mario_x_vel = struct.unpack('<b', payload[4:5])[0]    # Signed byte
        mario_y_vel = struct.unpack('<b', payload[5:6])[0]    # Signed byte
        
        power_state = payload[6]
        animation_state = payload[7]
        direction = payload[8]
        player_state = payload[9]
        lives = payload[10]
        invincibility_timer = payload[11]
        mario_x_raw = payload[12]
        crouching = payload[13]
        # bytes 14-15 are reserved
        
        game_state.update({
            'mario_x': mario_x_world,
            'mario_y': mario_y_level,
            'mario_x_vel': mario_x_vel,
            'mario_y_vel': mario_y_vel,
            'power_state': power_state,
            'animation_state': animation_state,
            'direction': direction,
            'player_state': player_state,
            'lives': lives,
            'invincible': invincibility_timer,
            'mario_x_raw': mario_x_raw,
            'crouching': crouching,
            'on_ground': 1 if mario_y_vel == 0 else 0,  # Approximate
            'facing_direction': direction,
            'velocity_magnitude': (mario_x_vel**2 + mario_y_vel**2)**0.5
        })
        
        # Enemy Data Block (32 bytes: positions 16-47)
        enemies = []
        enemy_count = 0
        closest_enemy_distance = 999.0
        
        for i in range(8):  # 8 enemies, 4 bytes each
            offset = 16 + i * 4
            enemy_type = payload[offset]
            enemy_x_pos = payload[offset + 1]
            enemy_y_pos = payload[offset + 2]
            enemy_state = payload[offset + 3]
            
            enemies.append({
                'type': enemy_type,
                'x_pos': enemy_x_pos,
                'y_pos': enemy_y_pos,
                'state': enemy_state,
                'is_active': enemy_type > 0
            })
            
            if enemy_type > 0:
                enemy_count += 1
                # Calculate distance to Mario
                distance = ((enemy_x_pos - mario_x_raw)**2 + (enemy_y_pos - mario_y_level)**2)**0.5
                closest_enemy_distance = min(closest_enemy_distance, distance)
        
        game_state.update({
            'enemies': enemies,
            'enemy_count': enemy_count,
            'closest_enemy_distance': closest_enemy_distance
        })
        
        # Level Data Block (64 bytes: positions 48-111)
        camera_x = struct.unpack('<H', payload[48:50])[0]  # Little-endian uint16
        world_number = payload[50]
        level_number = payload[51]
        score_100k = payload[52]
        score_10k = payload[53]
        score_1k = payload[54]
        score_100 = payload[55]
        
        time_remaining = struct.unpack('<I', payload[56:60])[0]  # Little-endian uint32
        total_coins = struct.unpack('<H', payload[60:62])[0]     # Little-endian uint16
        
        game_state.update({
            'camera_x': camera_x,
            'world_number': world_number,
            'level_number': level_number,
            'score_100k': score_100k,
            'score_10k': score_10k,
            'score_1k': score_1k,
            'score_100': score_100,
            'time_remaining': time_remaining,
            'total_coins': total_coins
        })
        
        # Enhanced features (if enabled and available in payload)
        if self.enhanced_features and len(payload) >= 112:
            # Power-up information (8 bytes: positions 62-69)
            powerup_type = payload[62]
            powerup_x_pos = payload[63]
            powerup_y_pos = payload[64]
            powerup_state = payload[65]
            powerup_world_x = struct.unpack('<H', payload[66:68])[0]
            powerup_is_active = payload[68] > 0
            
            powerup_distance = 999.0
            if powerup_is_active:
                powerup_distance = ((powerup_x_pos - mario_x_raw)**2 + (powerup_y_pos - mario_y_level)**2)**0.5
            
            game_state.update({
                'powerup_present': powerup_is_active,
                'powerup_distance': powerup_distance,
                'powerup_type': powerup_type
            })
            
            # Threat assessment (8 bytes: positions 70-77)
            threat_count = payload[70]
            threats_ahead = payload[71]
            threats_behind = payload[72]
            nearest_threat_distance = struct.unpack('<H', payload[74:76])[0]
            
            game_state.update({
                'threat_count': threat_count,
                'threats_ahead': threats_ahead,
                'threats_behind': threats_behind,
                'nearest_threat_distance': nearest_threat_distance
            })
            
            # Enhanced Mario velocity data (4 bytes: positions 78-81)
            enhanced_mario_x_vel = struct.unpack('<b', payload[78:79])[0]
            enhanced_mario_y_vel = struct.unpack('<b', payload[79:80])[0]
            mario_facing = payload[80]
            mario_below_viewport = payload[81]
            
            # Level tile sampling (16 bytes: positions 82-97)
            # Parse 4x4 grid of tiles around Mario
            solid_tiles_ahead = 0
            pit_detected = False
            
            for i in range(16):
                tile_value = payload[82 + i]
                # Check tiles ahead of Mario (right side of grid)
                if i % 4 >= 2:  # Right side of 4x4 grid
                    if tile_value > 0:
                        solid_tiles_ahead += 1
                    elif tile_value == 0 and i >= 8:  # Bottom rows with no tile = pit
                        pit_detected = True
            
            game_state.update({
                'solid_tiles_ahead': solid_tiles_ahead,
                'pit_detected': pit_detected,
                'mario_below_viewport': mario_below_viewport
            })
        else:
            # Default values for enhanced features when not available
            game_state.update({
                'powerup_present': False,
                'powerup_distance': 999.0,
                'solid_tiles_ahead': 0,
                'pit_detected': False,
                'mario_below_viewport': 0
            })
        
        # Game Variables Block (16 bytes: positions 112-127)
        game_engine_state = payload[112]
        level_progress_raw = payload[113]
        distance_to_flag = struct.unpack('<H', payload[114:116])[0]
        frame_id = struct.unpack('<I', payload[116:120])[0]
        timestamp = struct.unpack('<I', payload[120:124])[0]
        
        game_state.update({
            'game_engine_state': game_engine_state,
            'level_progress': level_progress_raw / 100.0,  # Convert from percentage
            'distance_to_flag': distance_to_flag,
            'frame_id': frame_id,
            'timestamp': timestamp
        })
        
        return game_state


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
    
    Combines frame preprocessing, state normalization, frame stacking,
    and binary payload parsing into a single convenient interface.
    """
    
    def __init__(
        self,
        stack_size: int = 4,
        frame_size: Tuple[int, int] = (84, 84),
        enhanced_features: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize Mario preprocessor.
        
        Args:
            stack_size: Number of frames to stack
            frame_size: Target frame dimensions
            enhanced_features: Whether to use enhanced 20-feature mode
            device: Device for tensor operations
        """
        self.enhanced_features = enhanced_features
        state_vector_size = 20 if enhanced_features else 12
        
        self.frame_stack = FrameStack(stack_size, frame_size, state_vector_size, device)
        self.frame_preprocessor = FramePreprocessor(frame_size, device=device)
        self.state_normalizer = StateNormalizer(enhanced_features)
        self.binary_parser = BinaryPayloadParser(enhanced_features)
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
    
    def process_binary_payload(
        self,
        raw_frame: Union[torch.Tensor, np.ndarray],
        binary_payload: bytes
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single step using binary payload from Lua script.
        
        Args:
            raw_frame: Raw game frame
            binary_payload: 128-byte binary payload from Lua
            
        Returns:
            Tuple of (stacked_frames, state_vector) ready for network input
        """
        # Parse binary payload to game state
        game_state = self.binary_parser.parse_payload(binary_payload)
        
        # Process normally
        return self.process_step(raw_frame, game_state)
    
    def get_current_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current stacked state without adding new frame.
        
        Returns:
            Current stacked frames and state vector with batch dimension
        """
        return self.frame_stack.get_batch_input()
    
    def get_state_vector_size(self) -> int:
        """Get the size of the state vector."""
        return self.state_normalizer.state_vector_size
    
    def is_enhanced_mode(self) -> bool:
        """Check if enhanced features are enabled."""
        return self.enhanced_features


class EnhancedFeatureValidator:
    """
    Validates enhanced feature data for consistency and correctness.
    
    Provides comprehensive validation for the new 20-feature state vector
    and binary payload data to ensure system reliability.
    """
    
    def __init__(self, enhanced_features: bool = False):
        """
        Initialize validator.
        
        Args:
            enhanced_features: Whether enhanced features are enabled
        """
        self.enhanced_features = enhanced_features
        self.validation_ranges = {
            # Basic features (0-11)
            'mario_x_norm': (0.0, 1.0),
            'mario_y_norm': (0.0, 1.0),
            'mario_x_vel_norm': (-1.0, 1.0),
            'mario_y_vel_norm': (-1.0, 1.0),
            'power_state_small': (0.0, 1.0),
            'power_state_big': (0.0, 1.0),
            'power_state_fire': (0.0, 1.0),
            'on_ground': (0.0, 1.0),
            'direction': (0.0, 1.0),
            'lives_norm': (0.0, 1.0),
            'invincible': (0.0, 1.0),
            'level_progress': (0.0, 1.0),
            # Enhanced features (12-19)
            'closest_enemy_norm': (0.0, 1.0),
            'enemy_count_norm': (0.0, 1.0),
            'powerup_present': (0.0, 1.0),
            'powerup_distance_norm': (0.0, 1.0),
            'solid_tiles_norm': (0.0, 1.0),
            'pit_detected': (0.0, 1.0),
            'velocity_mag_norm': (0.0, 1.0),
            'facing_direction': (0.0, 1.0)
        }
    
    def validate_state_vector(self, state_vector: torch.Tensor) -> Dict[str, Any]:
        """
        Validate a normalized state vector.
        
        Args:
            state_vector: Normalized state vector tensor
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        expected_size = 20 if self.enhanced_features else 12
        if state_vector.shape[-1] != expected_size:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Expected state vector size {expected_size}, got {state_vector.shape[-1]}"
            )
            return validation_result
        
        # Check value ranges
        state_np = state_vector.detach().cpu().numpy()
        feature_names = list(self.validation_ranges.keys())[:expected_size]
        
        for i, (feature_name, (min_val, max_val)) in enumerate(zip(feature_names,
                                                                  [self.validation_ranges[name] for name in feature_names])):
            value = state_np[i] if state_vector.dim() == 1 else state_np[0, i]
            
            if not (min_val <= value <= max_val):
                validation_result['warnings'].append(
                    f"Feature {feature_name}[{i}] = {value:.3f} outside expected range [{min_val}, {max_val}]"
                )
            
            validation_result['stats'][feature_name] = float(value)
        
        # Check for NaN or infinite values
        if torch.isnan(state_vector).any():
            validation_result['valid'] = False
            validation_result['errors'].append("State vector contains NaN values")
        
        if torch.isinf(state_vector).any():
            validation_result['valid'] = False
            validation_result['errors'].append("State vector contains infinite values")
        
        return validation_result
    
    def validate_binary_payload(self, payload: bytes) -> Dict[str, Any]:
        """
        Validate binary payload structure and content.
        
        Args:
            payload: Binary payload from Lua script
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'payload_info': {}
        }
        
        # Check payload size
        if len(payload) != 128:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Expected 128-byte payload, got {len(payload)} bytes"
            )
            return validation_result
        
        try:
            # Parse key fields for validation
            mario_x_world = struct.unpack('<H', payload[0:2])[0]
            mario_y_level = struct.unpack('<H', payload[2:4])[0]
            mario_x_vel = struct.unpack('<b', payload[4:5])[0]
            mario_y_vel = struct.unpack('<b', payload[5:6])[0]
            lives = payload[10]
            frame_id = struct.unpack('<I', payload[116:120])[0]
            
            validation_result['payload_info'] = {
                'mario_x_world': mario_x_world,
                'mario_y_level': mario_y_level,
                'mario_x_vel': mario_x_vel,
                'mario_y_vel': mario_y_vel,
                'lives': lives,
                'frame_id': frame_id
            }
            
            # Sanity checks
            if mario_x_world > 65535:
                validation_result['warnings'].append(f"Mario X position seems high: {mario_x_world}")
            
            if mario_y_level > 240:
                validation_result['warnings'].append(f"Mario Y position out of screen bounds: {mario_y_level}")
            
            if lives > 99:
                validation_result['warnings'].append(f"Lives count seems high: {lives}")
            
            if abs(mario_x_vel) > 127:
                validation_result['warnings'].append(f"Mario X velocity out of range: {mario_x_vel}")
            
            if abs(mario_y_vel) > 127:
                validation_result['warnings'].append(f"Mario Y velocity out of range: {mario_y_vel}")
                
        except struct.error as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Failed to parse payload: {e}")
        
        return validation_result
    
    def validate_game_state(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parsed game state dictionary.
        
        Args:
            game_state: Parsed game state dictionary
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'missing_keys': [],
            'extra_keys': []
        }
        
        # Required keys for basic mode
        required_keys = {
            'mario_x', 'mario_y', 'mario_x_vel', 'mario_y_vel', 'power_state',
            'direction', 'lives', 'on_ground'
        }
        
        # Additional required keys for enhanced mode
        if self.enhanced_features:
            required_keys.update({
                'closest_enemy_distance', 'enemy_count', 'powerup_present',
                'powerup_distance', 'solid_tiles_ahead', 'pit_detected',
                'velocity_magnitude', 'facing_direction'
            })
        
        # Check for missing keys
        missing_keys = required_keys - set(game_state.keys())
        if missing_keys:
            validation_result['missing_keys'] = list(missing_keys)
            validation_result['warnings'].append(f"Missing keys: {missing_keys}")
        
        # Check value ranges for key fields
        if 'mario_x' in game_state:
            if not (0 <= game_state['mario_x'] <= 65535):
                validation_result['warnings'].append(f"Mario X out of range: {game_state['mario_x']}")
        
        if 'mario_y' in game_state:
            if not (0 <= game_state['mario_y'] <= 240):
                validation_result['warnings'].append(f"Mario Y out of range: {game_state['mario_y']}")
        
        if 'lives' in game_state:
            if not (0 <= game_state['lives'] <= 99):
                validation_result['warnings'].append(f"Lives out of range: {game_state['lives']}")
        
        return validation_result


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
    
    # Test state normalizer (12-feature mode)
    print("\nTesting state normalizer (12-feature mode)...")
    state_normalizer = StateNormalizer(enhanced_features=False)
    
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
    
    # Test enhanced state normalizer (20-feature mode)
    print("\nTesting enhanced state normalizer (20-feature mode)...")
    enhanced_state_normalizer = StateNormalizer(enhanced_features=True)
    
    enhanced_game_state = dummy_game_state.copy()
    enhanced_game_state.update({
        'closest_enemy_distance': 150.0,
        'enemy_count': 2,
        'powerup_present': True,
        'powerup_distance': 80.0,
        'solid_tiles_ahead': 3,
        'pit_detected': False,
        'velocity_magnitude': 25.0,
        'facing_direction': 1
    })
    
    enhanced_normalized_state = enhanced_state_normalizer.normalize_state_vector(enhanced_game_state)
    print(f"Enhanced normalized state shape: {enhanced_normalized_state.shape}")
    print(f"Enhanced normalized state: {enhanced_normalized_state}")
    
    # Test binary payload parser
    print("\nTesting binary payload parser...")
    binary_parser = BinaryPayloadParser(enhanced_features=True)
    
    # Create dummy 128-byte payload
    dummy_payload = bytearray(128)
    # Mario data (first 16 bytes)
    dummy_payload[0:2] = struct.pack('<H', 1000)  # mario_x_world
    dummy_payload[2:4] = struct.pack('<H', 120)   # mario_y_level
    dummy_payload[4] = 20   # mario_x_vel (signed)
    dummy_payload[5] = 246  # mario_y_vel (-10 as unsigned byte)
    dummy_payload[6] = 1    # power_state
    dummy_payload[10] = 3   # lives
    
    parsed_state = binary_parser.parse_payload(bytes(dummy_payload))
    print(f"Parsed state keys: {list(parsed_state.keys())}")
    print(f"Mario X: {parsed_state['mario_x']}, Mario Y: {parsed_state['mario_y']}")
    print(f"Mario X Vel: {parsed_state['mario_x_vel']}, Mario Y Vel: {parsed_state['mario_y_vel']}")
    
    # Test frame stack with variable sizes
    print("\nTesting frame stack with variable state vector sizes...")
    frame_stack_12 = FrameStack(state_vector_size=12, device=device)
    frame_stack_20 = FrameStack(state_vector_size=20, device=device)
    
    # Add some frames
    for i in range(5):
        frame = torch.randn(84, 84)
        state_12 = torch.randn(12)
        state_20 = torch.randn(20)
        frame_stack_12.add_frame(frame, state_12)
        frame_stack_20.add_frame(frame, state_20)
    
    stacked_frames_12, current_state_12 = frame_stack_12.get_batch_input()
    stacked_frames_20, current_state_20 = frame_stack_20.get_batch_input()
    print(f"12-feature - Stacked frames: {stacked_frames_12.shape}, State: {current_state_12.shape}")
    print(f"20-feature - Stacked frames: {stacked_frames_20.shape}, State: {current_state_20.shape}")
    
    # Test complete preprocessors
    print("\nTesting complete Mario preprocessors...")
    mario_preprocessor_12 = MarioPreprocessor(enhanced_features=False, device=device)
    mario_preprocessor_20 = MarioPreprocessor(enhanced_features=True, device=device)
    
    # Process a step
    raw_frame = np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8)
    
    stacked_frames_12, state_vector_12 = mario_preprocessor_12.process_step(raw_frame, dummy_game_state)
    stacked_frames_20, state_vector_20 = mario_preprocessor_20.process_step(raw_frame, enhanced_game_state)
    
    print(f"12-feature preprocessor output:")
    print(f"  Stacked frames: {stacked_frames_12.shape}")
    print(f"  State vector: {state_vector_12.shape}")
    print(f"20-feature preprocessor output:")
    print(f"  Stacked frames: {stacked_frames_20.shape}")
    print(f"  State vector: {state_vector_20.shape}")
    
    # Test binary payload processing
    print("\nTesting binary payload processing...")
    stacked_frames_bin, state_vector_bin = mario_preprocessor_20.process_binary_payload(raw_frame, bytes(dummy_payload))
    print(f"Binary payload processing output:")
    print(f"  Stacked frames: {stacked_frames_bin.shape}")
    print(f"  State vector: {state_vector_bin.shape}")
    
    print("\nAll preprocessing tests completed successfully!")