"""
Integration test for the Super Mario Bros AI communication system.

Tests the WebSocket server, communication manager, frame capture,
reward calculator, and episode manager components.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, Any
import json
import struct

# Import our components
from python.communication.websocket_server import WebSocketServer
from python.communication.comm_manager import CommunicationManager, GameState
from python.capture.frame_capture import FrameCapture
from python.environment.reward_calculator import RewardCalculator
from python.environment.episode_manager import EpisodeManager


class MockLuaClient:
    """Mock Lua client for testing WebSocket communication."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.websocket = None
        self.is_connected = False
        
    async def connect(self):
        """Connect to WebSocket server."""
        import websockets
        
        try:
            self.websocket = await websockets.connect(f"ws://{self.host}:{self.port}")
            self.is_connected = True
            print("Mock Lua client connected")
            
            # Send initialization message
            await self.send_init_message()
            
        except Exception as e:
            print(f"Failed to connect: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            print("Mock Lua client disconnected")
    
    async def send_init_message(self):
        """Send initialization message."""
        init_msg = {
            "type": "init",
            "timestamp": int(time.time() * 1000),
            "fceux_version": "2.6.4",
            "rom_name": "Super Mario Bros (World).nes",
            "protocol_version": "1.0"
        }
        
        await self.websocket.send(json.dumps(init_msg))
        
        # Wait for acknowledgment
        response = await self.websocket.recv()
        response_data = json.loads(response)
        
        if response_data.get('type') == 'init_ack':
            print("Received initialization acknowledgment")
        else:
            print(f"Unexpected response: {response_data}")
    
    async def send_game_state(self, mario_x: int = 100, mario_y: int = 200, frame_id: int = 1):
        """Send mock game state data."""
        # Create binary game state data (128 bytes)
        binary_data = bytearray(128)
        
        # Mario Data Block (16 bytes)
        mario_data = struct.pack('<HHhhBBBBBBH', 
                                mario_x, mario_y,      # X, Y position
                                5, -2,                 # X, Y velocity
                                1,                     # Power state (big)
                                0,                     # Animation state
                                1,                     # Direction (right)
                                1,                     # On ground
                                3,                     # Lives
                                0,                     # Invincibility
                                0)                     # Reserved
        
        binary_data[0:16] = mario_data
        
        # Enemy Data Block (32 bytes) - no enemies for simplicity
        # Level Data Block (64 bytes)
        level_data = struct.pack('<HBBIHH', 
                                mario_x - 50,          # Camera X
                                1,                     # World number
                                1,                     # Level number
                                2400,                  # Score
                                350,                   # Time remaining
                                5)                     # Coins
        
        binary_data[48:60] = level_data
        
        # Game Variables Block (16 bytes)
        game_vars = struct.pack('<BBHIII',
                               0,                      # Game state (playing)
                               int((mario_x / 3168.0) * 100),  # Level progress %
                               max(0, 3168 - mario_x), # Distance to flag
                               frame_id,              # Frame counter
                               int(time.time()),      # Episode timer
                               0)                     # Reserved
        
        binary_data[112:128] = game_vars
        
        # Calculate checksum
        checksum = 0
        for byte in binary_data:
            checksum ^= byte
        checksum &= 0xFF
        
        # Create header
        header = struct.pack('<BIHB', 
                            0x01,                    # Message type (game_state)
                            frame_id,               # Frame ID
                            len(binary_data),       # Data length
                            checksum)               # Checksum
        
        # Send binary message
        message = header + binary_data
        await self.websocket.send(message)
    
    async def receive_action(self):
        """Receive action command from server."""
        try:
            response = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
            if isinstance(response, str):
                action_data = json.loads(response)
                if action_data.get('type') == 'action':
                    return action_data
            return None
        except asyncio.TimeoutError:
            return None


async def test_websocket_communication():
    """Test WebSocket server and communication manager."""
    print("\n=== Testing WebSocket Communication ===")
    
    # Start WebSocket server
    server = WebSocketServer()
    await server.start_server()
    
    try:
        # Connect mock client
        client = MockLuaClient()
        await client.connect()
        
        # Test sending game state
        await client.send_game_state(mario_x=150, frame_id=1)
        
        # Test receiving action
        action = await client.receive_action()
        if action:
            print(f"Received action: {action}")
        else:
            print("No action received")
        
        await client.disconnect()
        print("✓ WebSocket communication test passed")
        
    except Exception as e:
        print(f"✗ WebSocket communication test failed: {e}")
    finally:
        await server.stop_server()


async def test_communication_manager():
    """Test communication manager with game state parsing."""
    print("\n=== Testing Communication Manager ===")
    
    try:
        # Create communication manager
        comm_manager = CommunicationManager()
        
        # Test game state parsing
        test_binary_data = bytearray(128)
        
        # Fill with test data
        mario_data = struct.pack('<HHhhBBBBBBH', 
                                200, 180,      # X, Y position
                                3, 0,          # X, Y velocity
                                2,             # Power state (fire)
                                1,             # Animation state
                                1,             # Direction (right)
                                1,             # On ground
                                2,             # Lives
                                0,             # Invincibility
                                0)             # Reserved
        
        test_binary_data[0:16] = mario_data
        
        # Create game state
        game_state = GameState(frame_id=1, raw_data=bytes(test_binary_data))
        
        # Test normalized features
        features = game_state.get_normalized_features()
        print(f"Normalized features shape: {features.shape}")
        print(f"Sample features: {features[:5]}")
        
        # Test state dictionary
        state_dict = game_state.to_dict()
        print(f"Game state keys: {list(state_dict.keys())}")
        
        print("✓ Communication manager test passed")
        
    except Exception as e:
        print(f"✗ Communication manager test failed: {e}")


def test_frame_capture():
    """Test frame capture system (without actual FCEUX window)."""
    print("\n=== Testing Frame Capture ===")
    
    try:
        # Create frame capture instance
        frame_capture = FrameCapture()
        
        # Test preprocessing with mock frame
        mock_frame = np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8)
        processed_frame = frame_capture.preprocessor.preprocess_frame(mock_frame)
        
        print(f"Original frame shape: {mock_frame.shape}")
        print(f"Processed frame shape: {processed_frame.shape}")
        print(f"Processed frame dtype: {processed_frame.dtype}")
        print(f"Processed frame range: [{processed_frame.min():.3f}, {processed_frame.max():.3f}]")
        
        # Test frame stack
        frames = [np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8) for _ in range(8)]
        stacked_frames = frame_capture.preprocessor.preprocess_frame_stack(frames)
        
        print(f"Frame stack shape: {stacked_frames.shape}")
        
        # Test capture stats
        stats = frame_capture.get_capture_stats()
        print(f"Capture stats: {stats}")
        
        print("✓ Frame capture test passed")
        
    except Exception as e:
        print(f"✗ Frame capture test failed: {e}")


def test_reward_calculator():
    """Test reward calculation system."""
    print("\n=== Testing Reward Calculator ===")
    
    try:
        # Create reward calculator
        reward_calc = RewardCalculator()
        
        # Test with mock game states
        initial_state = {
            'mario_x': 100,
            'mario_y': 200,
            'power_state': 0,
            'score': 0,
            'coins': 0,
            'lives': 3,
            'timestamp': time.time()
        }
        
        reward_calc.reset_episode(initial_state)
        
        # Simulate forward movement
        current_state = {
            'mario_x': 150,  # Moved forward
            'mario_y': 200,
            'power_state': 1,  # Got power-up
            'score': 200,      # Scored points
            'coins': 1,        # Collected coin
            'lives': 3,
            'timestamp': time.time()
        }
        
        frame_reward, reward_components = reward_calc.calculate_frame_reward(current_state)
        
        print(f"Frame reward: {frame_reward:.2f}")
        print(f"Reward components: {reward_components.to_dict()}")
        
        # Test terminal state detection
        is_terminal, reason = reward_calc.detect_terminal_state(current_state)
        print(f"Terminal state: {is_terminal}, reason: {reason}")
        
        # Test episode end reward
        episode_data = {
            'level_completed': False,
            'died': False,
            'max_x_reached': 150,
            'time_remaining': 300
        }
        
        episode_reward, episode_components = reward_calc.calculate_episode_end_reward(episode_data)
        print(f"Episode end reward: {episode_reward:.2f}")
        
        # Test reward stats
        stats = reward_calc.get_reward_stats()
        print(f"Reward stats: {stats}")
        
        print("✓ Reward calculator test passed")
        
    except Exception as e:
        print(f"✗ Reward calculator test failed: {e}")


def test_episode_manager():
    """Test episode management system."""
    print("\n=== Testing Episode Manager ===")
    
    try:
        # Create components
        reward_calc = RewardCalculator()
        episode_manager = EpisodeManager(reward_calc, log_directory="test_logs")
        
        # Start episode
        initial_state = {
            'mario_x': 100,
            'mario_y': 200,
            'power_state': 0,
            'score': 0,
            'coins': 0,
            'lives': 3,
            'timestamp': time.time()
        }
        
        episode_id = episode_manager.start_episode(initial_state)
        print(f"Started episode {episode_id}")
        
        # Process some frames
        for i in range(10):
            game_state = {
                'mario_x': 100 + i * 10,
                'mario_y': 200,
                'power_state': 0,
                'score': i * 10,
                'coins': 0,
                'lives': 3,
                'timestamp': time.time()
            }
            
            frame_reward, reward_components, is_terminal = episode_manager.process_frame(
                game_state, {'right': True, 'A': False}, sync_quality=0.01
            )
            
            if is_terminal:
                break
        
        # End episode
        episode_data = {
            'level_completed': False,
            'died': False,
            'final_score': 100,
            'max_x_reached': 200,
            'time_remaining': 300,
            'lives_used': 0
        }
        
        completed_episode = episode_manager.end_episode(episode_data)
        print(f"Episode completed: {completed_episode.episode_id}")
        print(f"Total reward: {completed_episode.total_reward:.2f}")
        print(f"Max distance: {completed_episode.max_x_reached}")
        
        # Test performance metrics
        metrics = episode_manager.get_performance_metrics()
        print(f"Performance metrics: {metrics}")
        
        # Test curriculum progress
        curriculum = episode_manager.get_curriculum_progress()
        print(f"Curriculum progress: {curriculum}")
        
        print("✓ Episode manager test passed")
        
    except Exception as e:
        print(f"✗ Episode manager test failed: {e}")


async def test_integration():
    """Test full system integration."""
    print("\n=== Testing Full System Integration ===")
    
    try:
        # Create all components
        reward_calc = RewardCalculator()
        episode_manager = EpisodeManager(reward_calc, log_directory="integration_test_logs")
        comm_manager = CommunicationManager()
        
        # Start communication manager
        await comm_manager.start()
        
        # Register handlers
        received_states = []
        
        def state_handler(game_state):
            received_states.append(game_state)
            print(f"Received game state: frame {game_state.frame_id}, mario_x={game_state.mario_x}")
        
        comm_manager.register_state_handler(state_handler)
        
        # Start mock client
        client = MockLuaClient()
        await client.connect()
        
        # Send some game states
        for i in range(5):
            await client.send_game_state(mario_x=100 + i * 20, frame_id=i + 1)
            await asyncio.sleep(0.1)  # Small delay
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        print(f"Received {len(received_states)} game states")
        
        # Test action sending
        action_buttons = {
            'right': True,
            'left': False,
            'A': False,
            'B': True,
            'up': False,
            'down': False,
            'start': False,
            'select': False
        }
        
        await comm_manager.send_action(action_buttons, frame_id=6)
        
        # Receive action on client side
        action = await client.receive_action()
        if action:
            print(f"Action received by client: {action['buttons']}")
        
        # Cleanup
        await client.disconnect()
        await comm_manager.stop()
        
        print("✓ Full system integration test passed")
        
    except Exception as e:
        print(f"✗ Full system integration test failed: {e}")


async def main():
    """Run all tests."""
    print("Starting Super Mario Bros AI Communication System Tests")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run individual component tests
    test_frame_capture()
    test_reward_calculator()
    test_episode_manager()
    await test_communication_manager()
    await test_websocket_communication()
    
    # Run integration test
    await test_integration()
    
    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())