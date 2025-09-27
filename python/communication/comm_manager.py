"""
Communication manager for Super Mario Bros AI training system.

Manages WebSocket connections, message routing, binary game state parsing,
and frame synchronization between FCEUX and the Python trainer.
"""

import asyncio
import logging
import struct
import time
from collections import deque
from typing import Dict, Any, Optional, Callable, Tuple, List
import numpy as np

from .websocket_server import WebSocketServer


class GameState:
    """Represents parsed game state data."""
    
    def __init__(self, frame_id: int, raw_data: bytes):
        """
        Initialize game state from binary data.
        
        Args:
            frame_id: Frame identifier
            raw_data: Raw binary game state data
        """
        self.frame_id = frame_id
        self.timestamp = time.time()
        
        # Parse binary data according to protocol specification
        self._parse_binary_data(raw_data)
    
    def _parse_binary_data(self, data: bytes):
        """
        Parse binary game state data according to communication protocol.
        
        Binary Message Structure (128 bytes total):
        - Mario Data (16 bytes)
        - Enemy Data (32 bytes max, 4 bytes per enemy)
        - Level Data (64 bytes)
        - Game Variables (16 bytes)
        """
        if len(data) < 128:
            raise ValueError(f"Invalid game state data length: {len(data)}, expected 128 bytes")
        
        # Parse Mario Data Block (16 bytes)
        mario_data = struct.unpack('<HHhhBBBBBBH', data[0:16])
        self.mario_x = mario_data[0]          # X Position (world coordinates)
        self.mario_y = mario_data[1]          # Y Position (world coordinates)
        self.mario_x_vel = mario_data[2]      # X Velocity (signed)
        self.mario_y_vel = mario_data[3]      # Y Velocity (signed)
        self.power_state = mario_data[4]      # Power State (0=small, 1=big, 2=fire)
        self.animation_state = mario_data[5]  # Animation State
        self.direction = mario_data[6]        # Direction Facing (0=left, 1=right)
        self.on_ground = mario_data[7]        # On Ground Flag
        self.lives = mario_data[8]            # Lives Remaining
        self.invincibility = mario_data[9]    # Invincibility Timer
        # mario_data[10] is reserved
        
        # Parse Enemy Data Block (32 bytes, up to 8 enemies)
        self.enemies = []
        for i in range(8):
            offset = 16 + (i * 4)
            if offset + 4 <= len(data):
                enemy_data = struct.unpack('<BBBB', data[offset:offset+4])
                enemy_type, enemy_x, enemy_y, enemy_state = enemy_data
                
                if enemy_type > 0:  # 0 = no enemy
                    self.enemies.append({
                        'type': enemy_type,
                        'x': enemy_x,
                        'y': enemy_y,
                        'state': enemy_state
                    })
        
        # Parse Level Data Block (64 bytes)
        level_offset = 48
        level_data = struct.unpack('<HBBIHH', data[level_offset:level_offset+12])
        self.camera_x = level_data[0]         # Camera X Position
        self.world_number = level_data[1]     # World Number
        self.level_number = level_data[2]     # Level Number
        self.score = level_data[3]            # Score (BCD encoded)
        self.time_remaining = level_data[4]   # Time Remaining
        self.coins = level_data[5]            # Coins Collected
        
        # Level layout data (52 bytes of compressed tile data)
        self.level_layout = data[level_offset+12:level_offset+64]
        
        # Parse Game Variables Block (16 bytes)
        game_vars_offset = 112
        game_vars = struct.unpack('<BBHIII', data[game_vars_offset:game_vars_offset+16])
        self.game_state = game_vars[0]        # Game State (0=playing, 1=paused, 2=game_over)
        self.level_progress = game_vars[1]    # Level Progress Percentage
        self.distance_to_flag = game_vars[2]  # Distance to Flag
        self.frame_counter = game_vars[3]     # Frame Counter
        self.episode_timer = game_vars[4]     # Episode Timer
        # game_vars[5] is reserved
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert game state to dictionary."""
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'mario_x': self.mario_x,
            'mario_y': self.mario_y,
            'mario_x_vel': self.mario_x_vel,
            'mario_y_vel': self.mario_y_vel,
            'power_state': self.power_state,
            'animation_state': self.animation_state,
            'direction': self.direction,
            'on_ground': self.on_ground,
            'lives': self.lives,
            'invincibility': self.invincibility,
            'enemies': self.enemies,
            'camera_x': self.camera_x,
            'world_number': self.world_number,
            'level_number': self.level_number,
            'score': self.score,
            'time_remaining': self.time_remaining,
            'coins': self.coins,
            'game_state': self.game_state,
            'level_progress': self.level_progress,
            'distance_to_flag': self.distance_to_flag,
            'frame_counter': self.frame_counter,
            'episode_timer': self.episode_timer
        }
    
    def get_normalized_features(self) -> np.ndarray:
        """
        Get normalized feature vector for neural network input.
        
        Returns:
            Normalized feature vector
        """
        features = np.array([
            self.mario_x / 3168.0,              # Normalized X position (World 1-1 length)
            self.mario_y / 240.0,               # Normalized Y position (screen height)
            self.mario_x_vel / 127.0,           # Normalized X velocity
            self.mario_y_vel / 127.0,           # Normalized Y velocity
            float(self.power_state) / 2.0,      # Normalized power state
            float(self.on_ground),              # On ground flag
            float(self.direction),              # Direction (0 or 1)
            self.lives / 5.0,                   # Normalized lives (assume max 5)
            float(self.invincibility > 0),      # Invincibility flag
            self.level_progress / 100.0,        # Level progress percentage
            self.time_remaining / 400.0,        # Normalized time (max 400)
            self.coins / 100.0                  # Normalized coins (assume max 100)
        ])
        
        return features


class FrameSynchronizer:
    """Handles frame synchronization between game state and captured frames."""
    
    def __init__(self, buffer_size: int = 10):
        """
        Initialize frame synchronizer.
        
        Args:
            buffer_size: Size of synchronization buffers
        """
        self.buffer_size = buffer_size
        self.state_buffer = deque(maxlen=buffer_size)
        self.frame_buffer = deque(maxlen=buffer_size)
        self.sync_tolerance = 0.016  # 16ms tolerance (1 frame at 60 FPS)
        
        # Synchronization metrics
        self.sync_metrics = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'avg_sync_delay': 0.0,
            'max_sync_delay': 0.0
        }
    
    def add_game_state(self, game_state: GameState):
        """
        Add game state to synchronization buffer.
        
        Args:
            game_state: Game state data
        """
        self.state_buffer.append({
            'state': game_state,
            'timestamp': game_state.timestamp,
            'frame_id': game_state.frame_id
        })
    
    def add_captured_frame(self, frame: np.ndarray, timestamp: float):
        """
        Add captured frame to synchronization buffer.
        
        Args:
            frame: Captured frame data
            timestamp: Frame capture timestamp
        """
        self.frame_buffer.append({
            'frame': frame,
            'timestamp': timestamp
        })
    
    def get_synchronized_data(self) -> Optional[Tuple[GameState, np.ndarray, float]]:
        """
        Get synchronized game state and frame data.
        
        Returns:
            Tuple of (game_state, frame, sync_quality) or None if no match
        """
        if not self.state_buffer or not self.frame_buffer:
            return None
        
        # Get latest game state
        latest_state_entry = self.state_buffer[-1]
        latest_state = latest_state_entry['state']
        state_timestamp = latest_state_entry['timestamp']
        
        # Find best matching frame
        best_frame = None
        best_sync_quality = float('inf')
        
        for frame_entry in self.frame_buffer:
            sync_delay = abs(frame_entry['timestamp'] - state_timestamp)
            if sync_delay < best_sync_quality:
                best_sync_quality = sync_delay
                best_frame = frame_entry['frame']
        
        if best_frame is None:
            return None
        
        # Update metrics
        self.sync_metrics['total_syncs'] += 1
        if best_sync_quality <= self.sync_tolerance:
            self.sync_metrics['successful_syncs'] += 1
        
        self.sync_metrics['avg_sync_delay'] = (
            (self.sync_metrics['avg_sync_delay'] * (self.sync_metrics['total_syncs'] - 1) + best_sync_quality) /
            self.sync_metrics['total_syncs']
        )
        self.sync_metrics['max_sync_delay'] = max(self.sync_metrics['max_sync_delay'], best_sync_quality)
        
        return latest_state, best_frame, best_sync_quality
    
    def get_sync_metrics(self) -> Dict[str, Any]:
        """Get synchronization quality metrics."""
        if self.sync_metrics['total_syncs'] == 0:
            return {'no_data': True}
        
        success_rate = self.sync_metrics['successful_syncs'] / self.sync_metrics['total_syncs']
        
        return {
            'total_syncs': self.sync_metrics['total_syncs'],
            'success_rate': success_rate,
            'avg_sync_delay_ms': self.sync_metrics['avg_sync_delay'] * 1000,
            'max_sync_delay_ms': self.sync_metrics['max_sync_delay'] * 1000,
            'within_tolerance': success_rate > 0.9  # 90% success rate threshold
        }


class CommunicationManager:
    """
    Manages WebSocket communication and message routing for the AI training system.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Initialize communication manager.
        
        Args:
            host: WebSocket server host
            port: WebSocket server port
        """
        self.websocket_server = WebSocketServer(host, port)
        self.frame_synchronizer = FrameSynchronizer()
        
        # Message queues
        self.incoming_states = asyncio.Queue()
        self.outgoing_actions = asyncio.Queue()
        
        # Event handlers
        self.state_handlers: List[Callable] = []
        self.episode_handlers: List[Callable] = []
        self.error_handlers: List[Callable] = []
        
        # Connection state
        self.is_connected = False
        self.current_episode_id = None
        
        # Frame tracking
        self.last_frame_id = 0
        self.frame_desync_count = 0
        self.max_desync_tolerance = 5
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Register WebSocket handlers
        self._setup_websocket_handlers()
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket message handlers."""
        self.websocket_server.register_binary_handler(self._handle_game_state)
        self.websocket_server.register_json_handler('episode_event', self._handle_episode_event)
        self.websocket_server.register_json_handler('error', self._handle_client_error)
    
    async def start(self):
        """Start the communication manager."""
        self.logger.info("Starting communication manager")
        await self.websocket_server.start_server()
        
        # Start message processing tasks
        asyncio.create_task(self._process_outgoing_actions())
        
        self.logger.info("Communication manager started")
    
    async def stop(self):
        """Stop the communication manager."""
        self.logger.info("Stopping communication manager")
        await self.websocket_server.stop_server()
        self.is_connected = False
        self.logger.info("Communication manager stopped")
    
    # Event handler registration
    
    def register_state_handler(self, handler: Callable[[GameState], None]):
        """Register handler for game state updates."""
        self.state_handlers.append(handler)
    
    def register_episode_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register handler for episode events."""
        self.episode_handlers.append(handler)
    
    def register_error_handler(self, handler: Callable[[str, str], None]):
        """Register handler for error events."""
        self.error_handlers.append(handler)
    
    # Message handlers
    
    async def _handle_game_state(self, frame_id: int, binary_data: bytes):
        """
        Handle incoming binary game state data.
        
        Args:
            frame_id: Frame identifier
            binary_data: Binary game state payload
        """
        try:
            # Check for frame desynchronization
            if frame_id != self.last_frame_id + 1:
                self.frame_desync_count += 1
                self.logger.warning(f"Frame desync detected: expected {self.last_frame_id + 1}, got {frame_id}")
                
                if self.frame_desync_count > self.max_desync_tolerance:
                    await self.websocket_server._send_error(
                        "FRAME_DESYNC",
                        f"Frame desync count exceeded tolerance: {self.frame_desync_count}"
                    )
                    return
            else:
                self.frame_desync_count = 0  # Reset on successful sync
            
            self.last_frame_id = frame_id
            
            # Parse game state
            game_state = GameState(frame_id, binary_data)
            
            # Add to synchronization buffer
            self.frame_synchronizer.add_game_state(game_state)
            
            # Queue for processing
            await self.incoming_states.put(game_state)
            
            # Notify handlers
            for handler in self.state_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(game_state)
                    else:
                        handler(game_state)
                except Exception as e:
                    self.logger.error(f"Error in state handler: {e}")
            
        except Exception as e:
            self.logger.error(f"Error handling game state: {e}")
            await self.websocket_server._send_error("GAME_STATE_ERROR", str(e))
    
    async def _handle_episode_event(self, data: Dict[str, Any]):
        """Handle episode event messages."""
        event = data.get('event')
        episode_id = data.get('episode_id')
        
        self.logger.info(f"Episode {episode_id} event: {event}")
        
        # Update current episode
        if event in ['death', 'level_complete', 'time_up', 'manual_reset']:
            self.current_episode_id = None
        else:
            self.current_episode_id = episode_id
        
        # Notify handlers
        for handler in self.episode_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                self.logger.error(f"Error in episode handler: {e}")
    
    async def _handle_client_error(self, data: Dict[str, Any]):
        """Handle error messages from client."""
        error_code = data.get('error_code')
        message = data.get('message')
        
        self.logger.error(f"Client error [{error_code}]: {message}")
        
        # Notify handlers
        for handler in self.error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(error_code, message)
                else:
                    handler(error_code, message)
            except Exception as e:
                self.logger.error(f"Error in error handler: {e}")
    
    # Action sending
    
    async def send_action(self, action_buttons: Dict[str, bool], frame_id: Optional[int] = None):
        """
        Send action to game.
        
        Args:
            action_buttons: Button states
            frame_id: Frame ID for synchronization
        """
        await self.outgoing_actions.put((action_buttons, frame_id))
    
    async def _process_outgoing_actions(self):
        """Process outgoing action queue."""
        while True:
            try:
                action_buttons, frame_id = await self.outgoing_actions.get()
                await self.websocket_server.send_action(action_buttons, frame_id)
            except Exception as e:
                self.logger.error(f"Error processing outgoing action: {e}")
    
    # Training control
    
    async def start_episode(self, episode_id: int):
        """Start a new training episode."""
        await self.websocket_server.send_training_control('start', episode_id)
        self.current_episode_id = episode_id
    
    async def reset_episode(self, episode_id: int, level: str = "1-1"):
        """Reset to start of level."""
        await self.websocket_server.send_training_control('reset', episode_id, level)
        self.current_episode_id = episode_id
    
    async def pause_training(self):
        """Pause training."""
        await self.websocket_server.send_training_control('pause')
    
    async def stop_training(self):
        """Stop training."""
        await self.websocket_server.send_training_control('stop')
        self.current_episode_id = None
    
    # Data access methods
    
    async def get_latest_state(self) -> Optional[GameState]:
        """Get the latest game state."""
        try:
            return await asyncio.wait_for(self.incoming_states.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
    
    def add_captured_frame(self, frame: np.ndarray, timestamp: float):
        """Add captured frame for synchronization."""
        self.frame_synchronizer.add_captured_frame(frame, timestamp)
    
    def get_synchronized_data(self) -> Optional[Tuple[GameState, np.ndarray, float]]:
        """Get synchronized game state and frame data."""
        return self.frame_synchronizer.get_synchronized_data()
    
    # Status and metrics
    
    def is_client_connected(self) -> bool:
        """Check if client is connected."""
        return self.websocket_server.is_client_connected()
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        stats = self.websocket_server.get_connection_stats()
        stats.update({
            'frame_desync_count': self.frame_desync_count,
            'current_episode_id': self.current_episode_id,
            'sync_metrics': self.frame_synchronizer.get_sync_metrics()
        })
        return stats
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'incoming_queue_size': self.incoming_states.qsize(),
            'outgoing_queue_size': self.outgoing_actions.qsize(),
            'sync_metrics': self.frame_synchronizer.get_sync_metrics(),
            'frame_desync_count': self.frame_desync_count
        }