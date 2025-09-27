"""
Quick Test Script for Super Mario Bros AI Training System

This script provides a quick way to test and validate the system functionality
without running a full training session. It includes various test modes for
different aspects of the system.

Usage:
    python examples/quick_test.py [OPTIONS]

Examples:
    python examples/quick_test.py --mode basic
    python examples/quick_test.py --mode network --episodes 5
    python examples/quick_test.py --mode benchmark --duration 60
"""

import argparse
import asyncio
import logging
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import system components
from python.training.trainer import MarioTrainer
from python.logging.csv_logger import CSVLogger
from python.training.training_utils import SystemHealthMonitor
from python.utils.config_loader import ConfigLoader
from python.models.dueling_dqn import DuelingDQN
from python.agents.dqn_agent import DQNAgent
from python.communication.websocket_server import WebSocketServer
from python.capture.frame_capture import FrameCapture


class QuickTester:
    """Quick testing utility for the Mario AI system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/training_config.yaml"
        self.config_loader = ConfigLoader()
        self.config = None
        self.results = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration for testing."""
        try:
            self.config = self.config_loader.load_config(self.config_path)
            print(f"✓ Configuration loaded from {self.config_path}")
            return self.config
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            return {}
    
    async def test_basic_functionality(self) -> bool:
        """Test basic system functionality."""
        print("\n" + "="*50)
        print("BASIC FUNCTIONALITY TEST")
        print("="*50)
        
        try:
            # Test 1: Configuration loading
            print("\n[1/6] Testing configuration loading...")
            config = self.load_config()
            if not config:
                return False
            
            # Test 2: Neural network creation
            print("[2/6] Testing neural network creation...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = DuelingDQN(
                input_channels=4,
                num_actions=12,
                hidden_size=512
            ).to(device)
            print(f"✓ Neural network created on {device}")
            
            # Test 3: Agent initialization
            print("[3/6] Testing DQN agent initialization...")
            agent = DQNAgent(
                state_dim=(4, 84, 84),
                action_dim=12,
                learning_rate=0.001,
                device=device
            )
            print("✓ DQN agent initialized")
            
            # Test 4: Frame capture
            print("[4/6] Testing frame capture...")
            frame_capture = FrameCapture(frame_width=84, frame_height=84, frame_stack_size=4)
            test_frame = np.random.randint(0, 255, (84, 84), dtype=np.uint8)
            processed_frame = frame_capture.preprocess_frame(test_frame)
            print(f"✓ Frame capture working (output shape: {processed_frame.shape})")
            
            # Test 5: CSV logging
            print("[5/6] Testing CSV logging...")
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                csv_logger = CSVLogger(log_directory=temp_dir, session_id="quick_test")
                csv_logger.log_training_step(
                    episode=1, step=1, reward=1.0, total_reward=1.0, epsilon=1.0,
                    loss=0.5, q_values={'mean': 0.1, 'std': 0.05},
                    mario_state={'x': 32, 'y': 208, 'x_max': 32},
                    action_taken=1, processing_time_ms=15.2,
                    learning_rate=0.001, replay_buffer_size=100
                )
                csv_logger.close()
                print("✓ CSV logging working")
            
            # Test 6: System health monitoring
            print("[6/6] Testing system health monitoring...")
            health_monitor = SystemHealthMonitor()
            health_data = health_monitor.check_system_health()
            print(f"✓ System health monitoring (CPU: {health_data['cpu_percent']:.1f}%, "
                  f"Memory: {health_data['memory_usage_mb']:.0f}MB)")
            
            self.results['basic_functionality'] = True
            print("\n🎉 Basic functionality test PASSED!")
            return True
            
        except Exception as e:
            print(f"\n❌ Basic functionality test FAILED: {e}")
            self.results['basic_functionality'] = False
            return False
    
    async def test_network_communication(self, episodes: int = 3) -> bool:
        """Test network communication with mock client."""
        print("\n" + "="*50)
        print("NETWORK COMMUNICATION TEST")
        print("="*50)
        
        try:
            # Start WebSocket server
            print("\n[1/3] Starting WebSocket server...")
            server = WebSocketServer(host="localhost", port=8765)
            await server.start_server()
            print("✓ WebSocket server started on localhost:8765")
            
            # Create mock client
            print("[2/3] Testing client connection...")
            import websockets
            import json
            
            async def mock_client():
                try:
                    async with websockets.connect("ws://localhost:8765") as websocket:
                        print("✓ Mock client connected")
                        
                        # Send mock frame data
                        for episode in range(episodes):
                            for step in range(10):
                                # Mock game state
                                game_state = {
                                    'mario_x': 32 + step * 10,
                                    'mario_y': 208,
                                    'score': step * 100,
                                    'coins': step // 3,
                                    'lives': 3,
                                    'time_remaining': 400 - step,
                                    'level_complete': False,
                                    'death': False
                                }
                                
                                # Mock frame data
                                frame_data = np.random.randint(0, 255, (84, 84), dtype=np.uint8)
                                
                                message = {
                                    'type': 'frame_data',
                                    'frame_id': episode * 10 + step,
                                    'timestamp': int(time.time() * 1000),
                                    'game_state': game_state,
                                    'frame_data': frame_data.flatten().tolist()
                                }
                                
                                await websocket.send(json.dumps(message))
                                
                                # Wait for action response
                                try:
                                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                                    action_data = json.loads(response)
                                    if action_data.get('type') == 'action':
                                        print(f"  Episode {episode+1}, Step {step+1}: "
                                              f"Action {action_data.get('action', 0)} received")
                                except asyncio.TimeoutError:
                                    print(f"  Episode {episode+1}, Step {step+1}: No action received (timeout)")
                                
                                await asyncio.sleep(0.1)  # Simulate frame rate
                        
                        return True
                        
                except Exception as e:
                    print(f"❌ Mock client error: {e}")
                    return False
            
            # Run mock client
            print("[3/3] Running mock client simulation...")
            client_result = await mock_client()
            
            # Stop server
            await server.stop_server()
            
            if client_result:
                self.results['network_communication'] = True
                print("\n🎉 Network communication test PASSED!")
                return True
            else:
                self.results['network_communication'] = False
                print("\n❌ Network communication test FAILED!")
                return False
                
        except Exception as e:
            print(f"\n❌ Network communication test FAILED: {e}")
            self.results['network_communication'] = False
            return False
    
    async def test_training_loop(self, episodes: int = 2) -> bool:
        """Test training loop with mock data."""
        print("\n" + "="*50)
        print("TRAINING LOOP TEST")
        print("="*50)
        
        try:
            print(f"\n[1/3] Initializing trainer for {episodes} episodes...")
            
            # Create temporary config for testing
            import tempfile
            import yaml
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                test_config = {
                    'training': {
                        'learning_rate': 0.001,
                        'batch_size': 16,
                        'max_episodes': episodes,
                        'max_steps_per_episode': 50,
                        'warmup_episodes': 1,
                        'save_frequency': episodes,
                        'evaluation_frequency': episodes,
                        'epsilon_start': 1.0,
                        'epsilon_end': 0.1,
                        'epsilon_decay': 0.99,
                        'replay_buffer_size': 1000,
                        'target_update_frequency': 25,
                        'curriculum': {'enabled': False}
                    },
                    'performance': {
                        'device': 'cpu',
                        'mixed_precision': False,
                        'compile_model': False
                    },
                    'network': {
                        'host': 'localhost',
                        'port': 8766  # Different port to avoid conflicts
                    },
                    'rewards': {
                        'distance_reward_scale': 1.0,
                        'completion_reward': 1000.0,
                        'death_penalty': -100.0
                    },
                    'capture': {
                        'frame_width': 84,
                        'frame_height': 84,
                        'frame_stack_size': 4
                    }
                }
                yaml.dump(test_config, f)
                temp_config_path = f.name
            
            try:
                # Initialize trainer with mocked components
                print("[2/3] Testing trainer initialization...")
                
                # Mock the external dependencies
                from unittest.mock import patch, Mock, AsyncMock
                
                with patch('python.training.trainer.WebSocketServer') as mock_ws, \
                     patch('python.training.trainer.FrameCapture') as mock_fc:
                    
                    # Setup WebSocket server mock
                    mock_ws_instance = Mock()
                    mock_ws_instance.is_client_connected.return_value = False
                    mock_ws_instance.start_server = AsyncMock()
                    mock_ws_instance.stop_server = AsyncMock()
                    mock_ws.return_value = mock_ws_instance
                    
                    # Setup frame capture mock
                    mock_fc_instance = Mock()
                    mock_fc_instance.preprocess_frame.return_value = np.zeros((4, 84, 84), dtype=np.float32)
                    mock_fc.return_value = mock_fc_instance
                    
                    # Create trainer
                    trainer = MarioTrainer(temp_config_path)
                    print("✓ Trainer initialized successfully")
                    
                    # Test trainer status
                    status = trainer.get_training_status()
                    print(f"✓ Trainer status: {status['training_phase']}")
                    
                    print("[3/3] Testing training components...")
                    
                    # Test agent functionality
                    state = np.random.random((4, 84, 84)).astype(np.float32)
                    action = trainer.agent.select_action(state)
                    print(f"✓ Agent action selection: {action}")
                    
                    # Test reward calculation
                    game_state = {'mario_x': 100, 'mario_y': 208, 'score': 500}
                    prev_state = {'mario_x': 90, 'mario_y': 208, 'score': 400}
                    reward = trainer.reward_calculator.calculate_reward(game_state, prev_state, action)
                    print(f"✓ Reward calculation: {reward}")
                    
                    # Test episode management
                    episode_complete = trainer.episode_manager.is_episode_complete(game_state, 50)
                    print(f"✓ Episode management: complete={episode_complete}")
                    
                    self.results['training_loop'] = True
                    print("\n🎉 Training loop test PASSED!")
                    return True
                    
            finally:
                # Cleanup temporary config file
                os.unlink(temp_config_path)
                
        except Exception as e:
            print(f"\n❌ Training loop test FAILED: {e}")
            self.results['training_loop'] = False
            return False
    
    async def test_performance_benchmark(self, duration: int = 30) -> bool:
        """Run performance benchmark test."""
        print("\n" + "="*50)
        print("PERFORMANCE BENCHMARK TEST")
        print("="*50)
        
        try:
            print(f"\n[1/4] Running {duration}-second performance benchmark...")
            
            # Test neural network inference speed
            print("[2/4] Testing neural network inference speed...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = DuelingDQN(input_channels=4, num_actions=12, hidden_size=512).to(device)
            model.eval()
            
            # Warm up
            dummy_input = torch.randn(1, 4, 84, 84).to(device)
            for _ in range(10):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # Benchmark inference
            start_time = time.time()
            inference_count = 0
            
            while time.time() - start_time < duration / 4:
                with torch.no_grad():
                    _ = model(dummy_input)
                inference_count += 1
            
            inference_time = time.time() - start_time
            fps = inference_count / inference_time
            print(f"✓ Neural network inference: {fps:.1f} FPS")
            
            # Test frame processing speed
            print("[3/4] Testing frame processing speed...")
            frame_capture = FrameCapture(frame_width=84, frame_height=84, frame_stack_size=4)
            
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < duration / 4:
                test_frame = np.random.randint(0, 255, (84, 84), dtype=np.uint8)
                _ = frame_capture.preprocess_frame(test_frame)
                frame_count += 1
            
            processing_time = time.time() - start_time
            frame_fps = frame_count / processing_time
            print(f"✓ Frame processing: {frame_fps:.1f} FPS")
            
            # Test system monitoring
            print("[4/4] Testing system monitoring...")
            health_monitor = SystemHealthMonitor()
            
            start_time = time.time()
            monitor_count = 0
            
            while time.time() - start_time < duration / 4:
                _ = health_monitor.check_system_health()
                monitor_count += 1
                time.sleep(0.01)  # Small delay to simulate real monitoring
            
            monitoring_time = time.time() - start_time
            monitor_fps = monitor_count / monitoring_time
            print(f"✓ System monitoring: {monitor_fps:.1f} checks/second")
            
            # Performance summary
            print(f"\nPerformance Summary:")
            print(f"  Neural Network: {fps:.1f} FPS")
            print(f"  Frame Processing: {frame_fps:.1f} FPS")
            print(f"  System Monitoring: {monitor_fps:.1f} checks/sec")
            print(f"  Device: {device}")
            
            # Check if performance meets minimum requirements
            min_requirements = {
                'inference_fps': 60.0,  # Should handle 60 FPS game
                'frame_fps': 60.0,
                'monitor_fps': 10.0
            }
            
            performance_ok = (
                fps >= min_requirements['inference_fps'] and
                frame_fps >= min_requirements['frame_fps'] and
                monitor_fps >= min_requirements['monitor_fps']
            )
            
            if performance_ok:
                self.results['performance_benchmark'] = True
                print("\n🎉 Performance benchmark test PASSED!")
                return True
            else:
                self.results['performance_benchmark'] = False
                print("\n⚠️  Performance benchmark test completed with warnings!")
                print("System may be too slow for real-time training.")
                return False
                
        except Exception as e:
            print(f"\n❌ Performance benchmark test FAILED: {e}")
            self.results['performance_benchmark'] = False
            return False
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("QUICK TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)
        
        print(f"\nTests Run: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print(f"\nDetailed Results:")
        for test_name, result in self.results.items():
            status = "✓ PASS" if result else "❌ FAIL"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")
        
        if passed_tests == total_tests:
            print(f"\n🎉 ALL TESTS PASSED! System is ready for training.")
        else:
            print(f"\n⚠️  Some tests failed. Please check the issues above.")
        
        print("="*60)


async def main():
    """Main function for quick testing."""
    parser = argparse.ArgumentParser(description="Quick Test for Super Mario Bros AI Training System")
    parser.add_argument("--mode", choices=["basic", "network", "training", "benchmark", "all"], 
                       default="all", help="Test mode to run")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes for network/training tests")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds for benchmark test")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Initialize tester
    tester = QuickTester(config_path=args.config)
    
    print("🎮 Super Mario Bros AI Training System - Quick Test")
    print("="*60)
    
    # Run selected tests
    if args.mode == "basic" or args.mode == "all":
        await tester.test_basic_functionality()
    
    if args.mode == "network" or args.mode == "all":
        await tester.test_network_communication(episodes=args.episodes)
    
    if args.mode == "training" or args.mode == "all":
        await tester.test_training_loop(episodes=args.episodes)
    
    if args.mode == "benchmark" or args.mode == "all":
        await tester.test_performance_benchmark(duration=args.duration)
    
    # Print summary
    tester.print_summary()
    
    # Exit with appropriate code
    all_passed = all(tester.results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)