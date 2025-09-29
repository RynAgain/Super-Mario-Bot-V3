#!/usr/bin/env python3
"""
Test script for enhanced communication protocol.

This script tests the enhanced communication system with 20-feature state processing,
binary payload parsing, and enhanced reward calculation integration.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from python.communication.websocket_server import WebSocketServer
from python.communication.comm_manager import CommunicationManager
from python.utils.config_loader import ConfigLoader
from python.utils.preprocessing import BinaryPayloadParser, EnhancedFeatureValidator
from python.environment.reward_calculator import RewardCalculator


class EnhancedCommunicationTester:
    """Test enhanced communication protocol functionality."""
    
    def __init__(self):
        """Initialize the tester."""
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Load configuration
        try:
            config_loader = ConfigLoader()
            self.config = config_loader.load_all_configs()
            self.logger.info("Configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            self.config = {}
    
    def setup_logging(self):
        """Setup logging for testing."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('test_enhanced_communication.log')
            ]
        )
    
    def test_binary_payload_parser(self):
        """Test binary payload parsing functionality."""
        self.logger.info("Testing binary payload parser...")
        
        try:
            # Test basic mode parser
            basic_parser = BinaryPayloadParser(enhanced_features=False)
            
            # Test enhanced mode parser
            enhanced_parser = BinaryPayloadParser(enhanced_features=True)
            
            # Create test payload (128 bytes)
            test_payload = self.create_test_payload()
            
            # Test enhanced parsing
            parsed_state = enhanced_parser.parse_payload(test_payload)
            
            # Validate parsed state
            expected_keys = [
                'mario_x', 'mario_y', 'mario_x_vel', 'mario_y_vel', 'power_state',
                'lives', 'enemy_count', 'closest_enemy_distance'
            ]
            
            for key in expected_keys:
                if key not in parsed_state:
                    raise ValueError(f"Missing key in parsed state: {key}")
            
            self.logger.info(f"Binary payload parsing test passed - parsed {len(parsed_state)} state fields")
            return True
            
        except Exception as e:
            self.logger.error(f"Binary payload parsing test failed: {e}")
            return False
    
    def test_enhanced_feature_validator(self):
        """Test enhanced feature validation."""
        self.logger.info("Testing enhanced feature validator...")
        
        try:
            validator = EnhancedFeatureValidator(enhanced_features=True)
            
            # Test payload validation
            test_payload = self.create_test_payload()
            validation_result = validator.validate_binary_payload(test_payload)
            
            if not validation_result['valid']:
                self.logger.warning(f"Payload validation warnings: {validation_result['errors']}")
            
            # Test game state validation
            test_game_state = {
                'mario_x': 1000, 'mario_y': 120, 'mario_x_vel': 20, 'mario_y_vel': -10,
                'power_state': 1, 'lives': 3, 'enemy_count': 2, 'closest_enemy_distance': 150.0,
                'powerup_present': True, 'powerup_distance': 80.0, 'solid_tiles_ahead': 3,
                'pit_detected': False, 'velocity_magnitude': 25.0, 'facing_direction': 1
            }
            
            state_validation = validator.validate_game_state(test_game_state)
            
            self.logger.info("Enhanced feature validation test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Enhanced feature validation test failed: {e}")
            return False
    
    def test_enhanced_reward_calculator(self):
        """Test enhanced reward calculation."""
        self.logger.info("Testing enhanced reward calculator...")
        
        try:
            # Test basic reward calculator
            basic_calculator = RewardCalculator(enhanced_features=False)
            
            # Test enhanced reward calculator
            enhanced_calculator = RewardCalculator(enhanced_features=True)
            
            # Create test game states
            initial_state = {
                'mario_x': 1000, 'mario_y': 120, 'lives': 3, 'power_state': 0,
                'enemy_count': 2, 'closest_enemy_distance': 200.0
            }
            
            current_state = {
                'mario_x': 1050, 'mario_y': 120, 'lives': 3, 'power_state': 1,
                'enemy_count': 1, 'closest_enemy_distance': 150.0,
                'powerup_present': False, 'velocity_magnitude': 15.0
            }
            
            # Reset and calculate rewards
            enhanced_calculator.reset_episode(initial_state)
            reward, components = enhanced_calculator.calculate_frame_reward(current_state)
            
            self.logger.info(f"Enhanced reward calculation test passed - reward: {reward:.3f}")
            self.logger.info(f"Reward components: {components.to_dict()}")
            return True
            
        except Exception as e:
            self.logger.error(f"Enhanced reward calculation test failed: {e}")
            return False
    
    def test_websocket_server_enhanced_features(self):
        """Test WebSocket server enhanced features."""
        self.logger.info("Testing WebSocket server enhanced features...")
        
        try:
            # Test basic WebSocket server
            basic_server = WebSocketServer(enhanced_features=False)
            
            # Test enhanced WebSocket server
            enhanced_server = WebSocketServer(enhanced_features=True)
            
            # Test configuration
            if not enhanced_server.enhanced_features:
                raise ValueError("Enhanced features not enabled in WebSocket server")
            
            if enhanced_server.protocol_version != "1.1":
                raise ValueError(f"Expected protocol version 1.1, got {enhanced_server.protocol_version}")
            
            # Test statistics
            stats = enhanced_server.get_enhanced_stats()
            expected_stat_keys = [
                'total_enhanced_frames', 'successful_parses', 'validation_errors'
            ]
            
            for key in expected_stat_keys:
                if key not in stats:
                    raise ValueError(f"Missing enhanced statistic: {key}")
            
            self.logger.info("WebSocket server enhanced features test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"WebSocket server enhanced features test failed: {e}")
            return False
    
    def test_communication_manager_enhanced_features(self):
        """Test communication manager enhanced features."""
        self.logger.info("Testing communication manager enhanced features...")
        
        try:
            # Test basic communication manager
            basic_manager = CommunicationManager(enhanced_features=False)
            
            # Test enhanced communication manager
            enhanced_manager = CommunicationManager(enhanced_features=True)
            
            # Test configuration
            if not enhanced_manager.enhanced_features:
                raise ValueError("Enhanced features not enabled in communication manager")
            
            # Test components
            if not hasattr(enhanced_manager, 'mario_preprocessor'):
                raise ValueError("Mario preprocessor not initialized")
            
            if not hasattr(enhanced_manager, 'reward_calculator'):
                raise ValueError("Reward calculator not initialized")
            
            # Test performance metrics
            metrics = enhanced_manager.get_performance_metrics()
            
            if 'enhanced_features_enabled' not in metrics:
                raise ValueError("Enhanced features status not in performance metrics")
            
            if not metrics['enhanced_features_enabled']:
                raise ValueError("Enhanced features not reported as enabled")
            
            self.logger.info("Communication manager enhanced features test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Communication manager enhanced features test failed: {e}")
            return False
    
    def test_backward_compatibility(self):
        """Test backward compatibility with 12-feature mode."""
        self.logger.info("Testing backward compatibility...")
        
        try:
            # Test that basic mode still works
            basic_server = WebSocketServer(enhanced_features=False)
            basic_manager = CommunicationManager(enhanced_features=False)
            basic_parser = BinaryPayloadParser(enhanced_features=False)
            
            # Test protocol version compatibility
            if basic_server.protocol_version not in ["1.0", "1.1"]:
                raise ValueError(f"Basic server protocol version not compatible: {basic_server.protocol_version}")
            
            # Test that enhanced components can be disabled
            enhanced_server = WebSocketServer(enhanced_features=True)
            enhanced_server.set_enhanced_features(False)
            
            if enhanced_server.enhanced_features:
                raise ValueError("Failed to disable enhanced features")
            
            self.logger.info("Backward compatibility test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Backward compatibility test failed: {e}")
            return False
    
    def create_test_payload(self) -> bytes:
        """Create a test 128-byte payload for testing."""
        import struct
        
        payload = bytearray(128)
        
        # Mario data (16 bytes)
        payload[0:2] = struct.pack('<H', 1000)  # mario_x_world
        payload[2:4] = struct.pack('<H', 120)   # mario_y_level
        payload[4] = 20   # mario_x_vel
        payload[5] = 246  # mario_y_vel (-10 as unsigned)
        payload[6] = 1    # power_state
        payload[10] = 3   # lives
        
        # Enemy data (32 bytes starting at offset 16)
        payload[16] = 1   # enemy_type
        payload[17] = 150 # enemy_x_pos
        payload[18] = 120 # enemy_y_pos
        payload[19] = 1   # enemy_state
        
        # Level data (64 bytes starting at offset 48)
        payload[48:50] = struct.pack('<H', 800)  # camera_x
        payload[50] = 1   # world_number
        payload[51] = 1   # level_number
        
        # Game variables (16 bytes starting at offset 112)
        payload[112] = 0  # game_engine_state
        payload[113] = 50 # level_progress
        
        return bytes(payload)
    
    def run_all_tests(self):
        """Run all enhanced communication tests."""
        self.logger.info("Starting enhanced communication protocol tests...")
        
        tests = [
            ("Binary Payload Parser", self.test_binary_payload_parser),
            ("Enhanced Feature Validator", self.test_enhanced_feature_validator),
            ("Enhanced Reward Calculator", self.test_enhanced_reward_calculator),
            ("WebSocket Server Enhanced Features", self.test_websocket_server_enhanced_features),
            ("Communication Manager Enhanced Features", self.test_communication_manager_enhanced_features),
            ("Backward Compatibility", self.test_backward_compatibility)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Running test: {test_name}")
            self.logger.info(f"{'='*60}")
            
            try:
                if test_func():
                    self.logger.info(f"✓ {test_name} PASSED")
                    passed += 1
                else:
                    self.logger.error(f"✗ {test_name} FAILED")
                    failed += 1
            except Exception as e:
                self.logger.error(f"✗ {test_name} FAILED with exception: {e}")
                failed += 1
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"TEST RESULTS")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Passed: {passed}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Total:  {passed + failed}")
        
        if failed == 0:
            self.logger.info("🎉 ALL TESTS PASSED!")
            return True
        else:
            self.logger.error(f"❌ {failed} TESTS FAILED")
            return False


def main():
    """Main test function."""
    tester = EnhancedCommunicationTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ Enhanced communication protocol tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Enhanced communication protocol tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()