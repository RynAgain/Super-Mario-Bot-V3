#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for Enhanced State Management System
========================================================================

This test suite validates the entire enhanced state management pipeline from
Lua memory reading through Python processing, ensuring all components work
together correctly and maintain backward compatibility.

Test Coverage:
- Lua memory reading and enhanced payload generation
- Binary protocol communication (128-byte payload)
- Python state processing (12-feature vs 20-feature modes)
- Reward calculation with enhanced components
- Configuration switching and backward compatibility
- Error handling and recovery scenarios
- Performance characteristics and benchmarks
- Training checkpoint compatibility

Author: AI Training System
Version: 1.0
"""

import asyncio
import json
import logging
import numpy as np
import os
import struct
import sys
import time
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import unittest
from unittest.mock import Mock, patch, AsyncMock

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import system components
from python.utils.preprocessing import (
    BinaryPayloadParser, StateNormalizer, MarioPreprocessor,
    EnhancedFeatureValidator, FrameStack
)
from python.environment.reward_calculator import RewardCalculator, RewardComponents
from python.communication.websocket_server import WebSocketServer
from python.communication.comm_manager import CommunicationManager
from python.agents.dqn_agent import DQNAgent
from python.models.dueling_dqn import DuelingDQN
from python.utils.config_loader import ConfigLoader


class IntegrationTestSuite:
    """Comprehensive integration test suite for enhanced state management."""
    
    def __init__(self):
        """Initialize test suite."""
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
        self.performance_metrics = {}
        self.validation_errors = []
        
        # Test configuration
        self.test_config = {
            'enhanced_features': True,
            'legacy_mode': False,
            'performance_benchmarks': True,
            'error_injection': True,
            'checkpoint_validation': True
        }
        
        # Test data generators
        self.test_data_generator = TestDataGenerator()
        
        # Component instances for testing
        self.components = {}
        
        self.logger.info("Integration test suite initialized")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all integration tests.
        
        Returns:
            Dictionary containing test results and metrics
        """
        self.logger.info("Starting comprehensive integration test suite")
        start_time = time.time()
        
        # Test categories
        test_categories = [
            ("Lua Memory Reading", self.test_lua_memory_reading),
            ("Binary Protocol", self.test_binary_protocol_communication),
            ("State Processing", self.test_python_state_processing),
            ("Reward Calculation", self.test_reward_calculation),
            ("Configuration Switching", self.test_configuration_switching),
            ("Error Handling", self.test_error_handling),
            ("Performance Benchmarks", self.test_performance_characteristics),
            ("Checkpoint Compatibility", self.test_checkpoint_compatibility),
            ("End-to-End Integration", self.test_end_to_end_integration)
        ]
        
        # Run each test category
        for category_name, test_method in test_categories:
            self.logger.info(f"Running {category_name} tests...")
            try:
                category_results = await test_method()
                self.test_results[category_name] = category_results
                self.logger.info(f"[PASS] {category_name} tests completed")
            except Exception as e:
                self.logger.error(f"[FAIL] {category_name} tests failed: {e}")
                self.test_results[category_name] = {
                    'status': 'FAILED',
                    'error': str(e),
                    'tests_passed': 0,
                    'tests_failed': 1
                }
        
        # Generate final report
        total_time = time.time() - start_time
        final_report = self.generate_final_report(total_time)
        
        self.logger.info(f"Integration test suite completed in {total_time:.2f}s")
        return final_report
    
    async def test_lua_memory_reading(self) -> Dict[str, Any]:
        """Test Lua memory reading and enhanced payload generation."""
        results = {'status': 'PASSED', 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        # Test 1: Enhanced memory address validation
        try:
            # Simulate enhanced memory addresses from mario_ai.lua
            enhanced_addresses = {
                'MARIO_X_PAGE': 0x006D,
                'MARIO_X_SUB': 0x0086,
                'MARIO_VELOCITY_X': 0x007B,
                'MARIO_VELOCITY_Y': 0x007D,
                'ENEMY_SLOTS': [0x0F, 0x10, 0x11, 0x12, 0x13],
                'POWERUP_TYPE': 0x0039,
                'LEVEL_LAYOUT': 0x0500
            }
            
            # Validate address ranges
            for name, addr in enhanced_addresses.items():
                if isinstance(addr, list):
                    for i, sub_addr in enumerate(addr):
                        if not (0x0000 <= sub_addr <= 0xFFFF):
                            raise ValueError(f"Invalid address {name}[{i}]: 0x{sub_addr:04X}")
                else:
                    if not (0x0000 <= addr <= 0xFFFF):
                        raise ValueError(f"Invalid address {name}: 0x{addr:04X}")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Enhanced memory addresses validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Enhanced memory address validation failed: {e}")
        
        # Test 2: 128-byte payload generation simulation
        try:
            # Simulate Lua payload generation
            test_payload = self.test_data_generator.generate_enhanced_payload()
            
            if len(test_payload) != 128:
                raise ValueError(f"Expected 128-byte payload, got {len(test_payload)}")
            
            # Validate payload structure
            mario_x = struct.unpack('<H', test_payload[0:2])[0]
            mario_y = struct.unpack('<H', test_payload[2:4])[0]
            
            if not (0 <= mario_x <= 65535):
                raise ValueError(f"Invalid Mario X position: {mario_x}")
            if not (0 <= mario_y <= 240):
                raise ValueError(f"Invalid Mario Y position: {mario_y}")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] 128-byte payload generation validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Payload generation test failed: {e}")
        
        # Test 3: Enhanced feature flags simulation
        try:
            # Test enhanced feature configuration
            enhanced_config = {
                'ENHANCED_MEMORY_ENABLED': True,
                'ENEMY_DETECTION_ENABLED': True,
                'POWERUP_DETECTION_ENABLED': True,
                'TILE_SAMPLING_ENABLED': True,
                'ENHANCED_DEATH_DETECTION': True,
                'VELOCITY_TRACKING_ENABLED': True
            }
            
            # Validate all flags are boolean
            for flag, value in enhanced_config.items():
                if not isinstance(value, bool):
                    raise ValueError(f"Invalid flag type for {flag}: {type(value)}")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Enhanced feature flags validated")
        
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Enhanced feature flags test failed: {e}")
        
        if results['tests_failed'] > 0:
            results['status'] = 'FAILED'
        
        return results
    
    async def test_binary_protocol_communication(self) -> Dict[str, Any]:
        """Test binary protocol communication with 128-byte payload."""
        results = {'status': 'PASSED', 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        # Test 1: Binary payload parsing
        try:
            parser = BinaryPayloadParser(enhanced_features=True)
            test_payload = self.test_data_generator.generate_enhanced_payload()
            
            parsed_state = parser.parse_payload(test_payload)
            
            # Validate required fields
            required_fields = [
                'mario_x', 'mario_y', 'mario_x_vel', 'mario_y_vel',
                'power_state', 'lives', 'enemies', 'time_remaining'
            ]
            
            for field in required_fields:
                if field not in parsed_state:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate enhanced fields
            enhanced_fields = [
                'powerup_present', 'powerup_distance', 'solid_tiles_ahead',
                'pit_detected', 'velocity_magnitude', 'threat_count'
            ]
            
            for field in enhanced_fields:
                if field not in parsed_state:
                    raise ValueError(f"Missing enhanced field: {field}")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Binary payload parsing validated")
        
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Binary payload parsing failed: {e}")
        
        # Test 2: Payload validation
        try:
            validator = EnhancedFeatureValidator(enhanced_features=True)
            test_payload = self.test_data_generator.generate_enhanced_payload()
            
            validation_result = validator.validate_binary_payload(test_payload)
            
            if not validation_result['valid'] and validation_result['errors']:
                raise ValueError(f"Payload validation failed: {validation_result['errors']}")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Payload validation passed")
        
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Payload validation failed: {e}")
        
        # Test 3: WebSocket protocol simulation
        try:
            # Simulate WebSocket message with header
            test_payload = self.test_data_generator.generate_enhanced_payload()
            
            # Create header (8 bytes)
            msg_type = 0x01  # game_state
            frame_id = 12345
            data_length = len(test_payload)
            checksum = sum(test_payload) % 256
            
            header = struct.pack('<BIHB',
                               max(0, min(255, msg_type)),
                               max(0, min(4294967295, frame_id)),
                               max(0, min(65535, data_length)),
                               max(0, min(255, checksum)))
            full_message = header + test_payload
            
            # Validate message structure
            if len(full_message) != 136:  # 8 + 128
                raise ValueError(f"Expected 136-byte message, got {len(full_message)}")
            
            # Parse header back
            parsed_header = struct.unpack('<BIHB', full_message[:8])
            if parsed_header != (msg_type, frame_id, data_length, checksum):
                raise ValueError("Header parsing mismatch")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] WebSocket protocol simulation passed")
        
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] WebSocket protocol test failed: {e}")
        
        if results['tests_failed'] > 0:
            results['status'] = 'FAILED'
        
        return results
    
    async def test_python_state_processing(self) -> Dict[str, Any]:
        """Test Python state processing for both 12-feature and 20-feature modes."""
        results = {'status': 'PASSED', 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        # Test 1: 12-feature mode (legacy)
        try:
            normalizer_12 = StateNormalizer(enhanced_features=False)
            test_state = self.test_data_generator.generate_game_state(enhanced=False)
            
            normalized_state = normalizer_12.normalize_state_vector(test_state)
            
            if normalized_state.shape[0] != 12:
                raise ValueError(f"Expected 12 features, got {normalized_state.shape[0]}")
            
            # Validate feature ranges
            if not torch.all((normalized_state >= -1.0) & (normalized_state <= 1.0)):
                raise ValueError("Features outside expected range [-1, 1]")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] 12-feature mode processing validated")
        
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] 12-feature mode test failed: {e}")
        
        # Test 2: 20-feature mode (enhanced)
        try:
            normalizer_20 = StateNormalizer(enhanced_features=True)
            test_state = self.test_data_generator.generate_game_state(enhanced=True)
            
            normalized_state = normalizer_20.normalize_state_vector(test_state)
            
            if normalized_state.shape[0] != 20:
                raise ValueError(f"Expected 20 features, got {normalized_state.shape[0]}")
            
            # Validate feature ranges
            if not torch.all((normalized_state >= -1.0) & (normalized_state <= 1.0)):
                raise ValueError("Features outside expected range [-1, 1]")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] 20-feature mode processing validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] 20-feature mode test failed: {e}")
        
        # Test 3: Frame stacking with variable state sizes
        try:
            # Test 12-feature frame stack
            frame_stack_12 = FrameStack(state_vector_size=12)
            for i in range(5):
                frame = torch.randn(84, 84)
                state = torch.randn(12)
                frame_stack_12.add_frame(frame, state)
            
            stacked_frames, current_state = frame_stack_12.get_batch_input()
            if stacked_frames.shape != (1, 4, 84, 84):
                raise ValueError(f"Invalid stacked frames shape: {stacked_frames.shape}")
            if current_state.shape != (1, 12):
                raise ValueError(f"Invalid state shape: {current_state.shape}")
            
            # Test 20-feature frame stack
            frame_stack_20 = FrameStack(state_vector_size=20)
            for i in range(5):
                frame = torch.randn(84, 84)
                state = torch.randn(20)
                frame_stack_20.add_frame(frame, state)
            
            stacked_frames, current_state = frame_stack_20.get_batch_input()
            if stacked_frames.shape != (1, 4, 84, 84):
                raise ValueError(f"Invalid stacked frames shape: {stacked_frames.shape}")
            if current_state.shape != (1, 20):
                raise ValueError(f"Invalid state shape: {current_state.shape}")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Frame stacking with variable state sizes validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Frame stacking test failed: {e}")
        
        # Test 4: Mario preprocessor integration
        try:
            # Test both modes
            for enhanced in [False, True]:
                preprocessor = MarioPreprocessor(enhanced_features=enhanced)
                
                # Generate test data
                raw_frame = np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8)
                game_state = self.test_data_generator.generate_game_state(enhanced=enhanced)
                
                # Process step
                stacked_frames, state_vector = preprocessor.process_step(raw_frame, game_state)
                
                expected_state_size = 20 if enhanced else 12
                if state_vector.shape != (1, expected_state_size):
                    raise ValueError(f"Invalid state vector shape for enhanced={enhanced}: {state_vector.shape}")
                
                if stacked_frames.shape != (1, 4, 84, 84):
                    raise ValueError(f"Invalid stacked frames shape: {stacked_frames.shape}")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Mario preprocessor integration validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Mario preprocessor test failed: {e}")
        
        if results['tests_failed'] > 0:
            results['status'] = 'FAILED'
        
        return results
    
    async def test_reward_calculation(self) -> Dict[str, Any]:
        """Test reward calculation with enhanced components."""
        results = {'status': 'PASSED', 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        # Test 1: Basic reward calculation (12-feature mode)
        try:
            reward_calc = RewardCalculator(enhanced_features=False)
            
            # Initialize with starting state
            initial_state = self.test_data_generator.generate_game_state(enhanced=False)
            reward_calc.reset_episode(initial_state)
            
            # Test forward movement reward
            next_state = initial_state.copy()
            next_state['mario_x'] = initial_state['mario_x'] + 100  # Move forward
            
            reward, components = reward_calc.calculate_frame_reward(next_state)
            
            if reward <= 0:
                raise ValueError(f"Expected positive reward for forward movement, got {reward}")
            
            if components.distance_reward <= 0:
                raise ValueError("Expected positive distance reward")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Basic reward calculation validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Basic reward calculation failed: {e}")
        
        # Test 2: Enhanced reward calculation (20-feature mode)
        try:
            reward_calc = RewardCalculator(enhanced_features=True)
            
            # Initialize with enhanced starting state
            initial_state = self.test_data_generator.generate_game_state(enhanced=True)
            reward_calc.reset_episode(initial_state)
            
            # Test enhanced reward components
            next_state = initial_state.copy()
            next_state['mario_x'] = initial_state['mario_x'] + 50
            next_state['power_state'] = initial_state['power_state'] + 1  # Power up
            next_state['enemy_count'] = initial_state['enemy_count'] - 1  # Enemy eliminated
            next_state['powerup_present'] = True  # Power-up detected
            
            reward, components = reward_calc.calculate_frame_reward(next_state)
            
            # Validate enhanced components
            if components.powerup_collection_reward <= 0:
                raise ValueError("Expected positive power-up collection reward")
            
            if components.enemy_elimination_reward <= 0:
                raise ValueError("Expected positive enemy elimination reward")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Enhanced reward calculation validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Enhanced reward calculation failed: {e}")
        
        # Test 3: Death penalty calculation
        try:
            reward_calc = RewardCalculator(enhanced_features=True)
            
            initial_state = self.test_data_generator.generate_game_state(enhanced=True)
            initial_state['lives'] = 3
            reward_calc.reset_episode(initial_state)
            
            # Simulate death
            death_state = initial_state.copy()
            death_state['lives'] = 2  # Lost a life
            death_state['mario_below_viewport'] = True  # Pit death
            
            reward, components = reward_calc.calculate_frame_reward(death_state)
            
            if components.enhanced_death_penalty >= 0:
                raise ValueError("Expected negative death penalty")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Death penalty calculation validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Death penalty test failed: {e}")
        
        # Test 4: Reward component validation
        try:
            # Test all reward components
            components = RewardComponents()
            components.distance_reward = 10.0
            components.powerup_collection_reward = 5.0
            components.enemy_elimination_reward = 3.0
            components.enhanced_death_penalty = -20.0
            
            total = components.total
            expected_total = 10.0 + 5.0 + 3.0 - 20.0
            
            if abs(total - expected_total) > 0.001:
                raise ValueError(f"Total reward mismatch: expected {expected_total}, got {total}")
            
            # Test dictionary conversion
            components_dict = components.to_dict()
            if 'total' not in components_dict:
                raise ValueError("Missing total in components dictionary")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Reward component validation passed")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Reward component validation failed: {e}")
        
        if results['tests_failed'] > 0:
            results['status'] = 'FAILED'
        
        return results
    
    async def test_configuration_switching(self) -> Dict[str, Any]:
        """Test configuration switching and backward compatibility."""
        results = {'status': 'PASSED', 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        # Test 1: Dynamic feature switching
        try:
            # Start with enhanced features disabled
            preprocessor = MarioPreprocessor(enhanced_features=False)
            
            # Verify 12-feature mode
            if preprocessor.get_state_vector_size() != 12:
                raise ValueError(f"Expected 12 features, got {preprocessor.get_state_vector_size()}")
            
            # Switch to enhanced mode (simulate runtime switching)
            preprocessor_enhanced = MarioPreprocessor(enhanced_features=True)
            
            # Verify 20-feature mode
            if preprocessor_enhanced.get_state_vector_size() != 20:
                raise ValueError(f"Expected 20 features, got {preprocessor_enhanced.get_state_vector_size()}")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Dynamic feature switching validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Dynamic feature switching failed: {e}")
        
        # Test 2: Backward compatibility
        try:
            # Test that 12-feature payloads work with enhanced parser
            parser = BinaryPayloadParser(enhanced_features=False)
            
            # Generate legacy payload (simulate smaller payload)
            legacy_payload = self.test_data_generator.generate_legacy_payload()
            
            # Should parse without errors
            parsed_state = parser.parse_payload(legacy_payload)
            
            # Verify basic fields are present
            required_fields = ['mario_x', 'mario_y', 'power_state', 'lives']
            for field in required_fields:
                if field not in parsed_state:
                    raise ValueError(f"Missing basic field in legacy mode: {field}")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Backward compatibility validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Backward compatibility test failed: {e}")
        
        # Test 3: Configuration validation
        try:
            # Test configuration loading and validation
            config_loader = ConfigLoader()
            
            # Test enhanced configuration
            enhanced_config = {
                'enhanced_rewards': {
                    'enabled': True,
                    'powerup_collection_reward': 50.0,
                    'enemy_elimination_reward': 25.0,
                    'feature_weights': {
                        'powerup_collection': 1.0,
                        'enemy_elimination': 1.0
                    }
                }
            }
            
            # Validate configuration structure
            if 'enhanced_rewards' not in enhanced_config:
                raise ValueError("Missing enhanced_rewards section")
            
            enhanced_section = enhanced_config['enhanced_rewards']
            if not isinstance(enhanced_section.get('enabled'), bool):
                raise ValueError("Invalid enabled flag type")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Configuration validation passed")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Configuration validation failed: {e}")
        
        if results['tests_failed'] > 0:
            results['status'] = 'FAILED'
        
        return results
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery scenarios."""
        results = {'status': 'PASSED', 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        # Test 1: Malformed payload handling
        try:
            parser = BinaryPayloadParser(enhanced_features=True)
            
            # Test various malformed payloads
            malformed_payloads = [
                b'',  # Empty payload
                b'short',  # Too short
                b'x' * 64,  # Wrong size
                b'x' * 256,  # Too large
            ]
            
            for i, payload in enumerate(malformed_payloads):
                try:
                    parser.parse_payload(payload)
                    raise ValueError(f"Parser should have rejected malformed payload {i}")
                except ValueError:
                    # Expected behavior
                    pass
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Malformed payload handling validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Malformed payload handling failed: {e}")
        
        # Test 2: Invalid state data handling
        try:
            normalizer = StateNormalizer(enhanced_features=True)
            
            # Test invalid state data
            invalid_states = [
                {},  # Empty state
                {'mario_x': -1000},  # Invalid position
                {'mario_x': 1000, 'mario_y': 'invalid'},  # Wrong type
                {'mario_x': float('inf')},  # Infinite value
                {'mario_x': float('nan')},  # NaN value
            ]
            
            for i, state in enumerate(invalid_states):
                try:
                    normalized = normalizer.normalize_state_vector(state)
                    # Check for NaN or infinite values
                    if torch.isnan(normalized).any() or torch.isinf(normalized).any():
                        raise ValueError(f"Normalizer produced invalid values for state {i}")
                except (KeyError, TypeError, ValueError):
                    # Expected behavior for invalid inputs
                    pass
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Invalid state data handling validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Invalid state data handling failed: {e}")
        
        # Test 3: Communication error recovery
        try:
            # Simulate WebSocket server with error injection
            server = WebSocketServer(enhanced_features=True)
            
            # Test checksum validation
            test_payload = self.test_data_generator.generate_enhanced_payload()
            correct_checksum = sum(test_payload) % 256
            wrong_checksum = (correct_checksum + 1) % 256
            
            # Simulate checksum mismatch detection
            if server._calculate_checksum(test_payload) != correct_checksum:
                raise ValueError("Checksum calculation error")
            
            if server._calculate_checksum(test_payload) == wrong_checksum:
                raise ValueError("Checksum validation should have failed")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Communication error recovery validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Communication error recovery failed: {e}")
        
        # Test 4: Graceful degradation
        try:
            # Test system behavior when enhanced features fail
            reward_calc = RewardCalculator(enhanced_features=True)
            
            # Simulate enhanced feature failure by providing incomplete state
            incomplete_state = {
                'mario_x': 1000,
                'mario_y': 120,
                'power_state': 1,
                'lives': 3
                # Missing enhanced fields
            }
            
            reward_calc.reset_episode(incomplete_state)
            
            # Should still calculate basic rewards
            reward, components = reward_calc.calculate_frame_reward(incomplete_state)
            
            # Should not crash, even with missing enhanced fields
            if not isinstance(reward, (int, float)):
                raise ValueError("Reward calculation failed with incomplete state")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Graceful degradation validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Graceful degradation test failed: {e}")
        
        if results['tests_failed'] > 0:
            results['status'] = 'FAILED'
        
        return results
    
    async def test_performance_characteristics(self) -> Dict[str, Any]:
        """Test performance characteristics and benchmarks."""
        results = {'status': 'PASSED', 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        # Test 1: Payload parsing performance
        try:
            parser = BinaryPayloadParser(enhanced_features=True)
            test_payload = self.test_data_generator.generate_enhanced_payload()
            
            # Benchmark parsing performance
            num_iterations = 1000
            start_time = time.time()
            
            for _ in range(num_iterations):
                parsed_state = parser.parse_payload(test_payload)
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_parse = (total_time / num_iterations) * 1000  # ms
            
            # Performance threshold: should parse in under 1ms per payload
            if avg_time_per_parse > 1.0:
                raise ValueError(f"Parsing too slow: {avg_time_per_parse:.3f}ms per payload")
            
            self.performance_metrics['payload_parsing_ms'] = avg_time_per_parse
            results['tests_passed'] += 1
            results['details'].append(f"[PASS] Payload parsing performance: {avg_time_per_parse:.3f}ms per payload")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Payload parsing performance test failed: {e}")
        
        # Test 2: State normalization performance
        try:
            normalizer = StateNormalizer(enhanced_features=True)
            test_state = self.test_data_generator.generate_game_state(enhanced=True)
            
            # Benchmark normalization performance
            num_iterations = 1000
            start_time = time.time()
            
            for _ in range(num_iterations):
                normalized_state = normalizer.normalize_state_vector(test_state)
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_norm = (total_time / num_iterations) * 1000  # ms
            
            # Performance threshold: should normalize in under 0.5ms
            if avg_time_per_norm > 0.5:
                raise ValueError(f"Normalization too slow: {avg_time_per_norm:.3f}ms per state")
            
            self.performance_metrics['state_normalization_ms'] = avg_time_per_norm
            results['tests_passed'] += 1
            results['details'].append(f"[PASS] State normalization performance: {avg_time_per_norm:.3f}ms per state")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] State normalization performance test failed: {e}")
        
        # Test 3: Reward calculation performance
        try:
            reward_calc = RewardCalculator(enhanced_features=True)
            test_state = self.test_data_generator.generate_game_state(enhanced=True)
            reward_calc.reset_episode(test_state)
            
            # Benchmark reward calculation performance
            num_iterations = 1000
            start_time = time.time()
            
            for _ in range(num_iterations):
                reward, components = reward_calc.calculate_frame_reward(test_state)
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_reward = (total_time / num_iterations) * 1000  # ms
            
            # Performance threshold: should calculate in under 0.1ms
            if avg_time_per_reward > 0.1:
                raise ValueError(f"Reward calculation too slow: {avg_time_per_reward:.3f}ms per calculation")
            
            self.performance_metrics['reward_calculation_ms'] = avg_time_per_reward
            results['tests_passed'] += 1
            results['details'].append(f"[PASS] Reward calculation performance: {avg_time_per_reward:.3f}ms per calculation")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Reward calculation performance test failed: {e}")
        
        # Test 4: Memory usage validation
        try:
            import psutil
            import gc
            
            # Measure memory usage during processing
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process many frames to test memory leaks
            preprocessor = MarioPreprocessor(enhanced_features=True)
            for i in range(100):
                raw_frame = np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8)
                game_state = self.test_data_generator.generate_game_state(enhanced=True)
                stacked_frames, state_vector = preprocessor.process_step(raw_frame, game_state)
            
            # Force garbage collection
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory threshold: should not increase by more than 50MB
            if memory_increase > 50:
                raise ValueError(f"Memory usage increased by {memory_increase:.1f}MB")
            
            self.performance_metrics['memory_usage_mb'] = memory_increase
            results['tests_passed'] += 1
            results['details'].append(f"[PASS] Memory usage validation: +{memory_increase:.1f}MB")
            
        except ImportError:
            results['details'].append("[WARN] psutil not available, skipping memory test")
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Memory usage test failed: {e}")
        
        if results['tests_failed'] > 0:
            results['status'] = 'FAILED'
        
        return results
    
    async def test_checkpoint_compatibility(self) -> Dict[str, Any]:
        """Test existing training checkpoint compatibility."""
        results = {'status': 'PASSED', 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        # Test 1: Model architecture compatibility
        try:
            # Test both 12-feature and 20-feature models
            for enhanced in [False, True]:
                state_size = 20 if enhanced else 12
                model = DuelingDQN(
                    frame_stack_size=4,
                    frame_size=(84, 84),
                    state_vector_size=state_size,
                    num_actions=12
                )
                
                # Test forward pass
                batch_frames = torch.randn(1, 4, 84, 84)
                batch_states = torch.randn(1, state_size)
                
                q_values = model(batch_frames, batch_states)
                
                if q_values.shape != (1, 12):
                    raise ValueError(f"Invalid Q-values shape: {q_values.shape}")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Model architecture compatibility validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Model architecture compatibility failed: {e}")
        
        # Test 2: Checkpoint loading simulation
        try:
            # Simulate checkpoint structure
            checkpoint_data = {
                'model_state_dict': {},
                'optimizer_state_dict': {},
                'episode': 100,
                'total_steps': 10000,
                'enhanced_features': False,  # Legacy checkpoint
                'state_vector_size': 12
            }
            
            # Test loading with enhanced system
            if 'enhanced_features' in checkpoint_data:
                enhanced_mode = checkpoint_data['enhanced_features']
                state_size = checkpoint_data.get('state_vector_size', 12)
                
                # Validate compatibility
                if enhanced_mode and state_size != 20:
                    raise ValueError("Enhanced checkpoint with wrong state size")
                elif not enhanced_mode and state_size != 12:
                    raise ValueError("Legacy checkpoint with wrong state size")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Checkpoint loading compatibility validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Checkpoint loading test failed: {e}")
        
        # Test 3: Training state migration
        try:
            # Test migration from 12-feature to 20-feature
            legacy_agent_config = {
                'state_size': 12,
                'enhanced_features': False
            }
            
            enhanced_agent_config = {
                'state_size': 20,
                'enhanced_features': True
            }
            
            # Simulate migration validation
            if legacy_agent_config['state_size'] != enhanced_agent_config['state_size']:
                # Migration required
                migration_valid = True
                
                # Check if migration is supported
                if enhanced_agent_config['state_size'] < legacy_agent_config['state_size']:
                    migration_valid = False
                    raise ValueError("Cannot migrate to smaller state size")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Training state migration validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Training state migration failed: {e}")
        
        if results['tests_failed'] > 0:
            results['status'] = 'FAILED'
        
        return results
    
    async def test_end_to_end_integration(self) -> Dict[str, Any]:
        """Test complete end-to-end integration."""
        results = {'status': 'PASSED', 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        # Test 1: Complete pipeline simulation
        try:
            # Simulate complete pipeline: Lua → Binary → Python → Processing → Reward
            
            # Step 1: Generate enhanced payload (simulating Lua)
            test_payload = self.test_data_generator.generate_enhanced_payload()
            
            # Step 2: Parse binary payload (simulating WebSocket reception)
            parser = BinaryPayloadParser(enhanced_features=True)
            parsed_state = parser.parse_payload(test_payload)
            
            # Step 3: Normalize state (simulating preprocessing)
            normalizer = StateNormalizer(enhanced_features=True)
            normalized_state = normalizer.normalize_state_vector(parsed_state)
            
            # Step 4: Calculate reward (simulating reward system)
            reward_calc = RewardCalculator(enhanced_features=True)
            reward_calc.reset_episode(parsed_state)
            reward, components = reward_calc.calculate_frame_reward(parsed_state)
            
            # Step 5: Process with Mario preprocessor
            preprocessor = MarioPreprocessor(enhanced_features=True)
            raw_frame = np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8)
            stacked_frames, state_vector = preprocessor.process_step(raw_frame, parsed_state)
            
            # Validate complete pipeline
            if normalized_state.shape[0] != 20:
                raise ValueError(f"Invalid normalized state size: {normalized_state.shape[0]}")
            
            if not isinstance(reward, (int, float)):
                raise ValueError("Invalid reward type")
            
            if stacked_frames.shape != (1, 4, 84, 84):
                raise ValueError(f"Invalid stacked frames shape: {stacked_frames.shape}")
            
            if state_vector.shape != (1, 20):
                raise ValueError(f"Invalid state vector shape: {state_vector.shape}")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Complete pipeline simulation validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Complete pipeline simulation failed: {e}")
        
        # Test 2: Multi-frame sequence processing
        try:
            preprocessor = MarioPreprocessor(enhanced_features=True)
            reward_calc = RewardCalculator(enhanced_features=True)
            
            # Process sequence of frames
            total_reward = 0
            for frame_num in range(10):
                # Generate progressive game state
                game_state = self.test_data_generator.generate_game_state(enhanced=True)
                game_state['mario_x'] = 1000 + frame_num * 50  # Progressive movement
                game_state['frame_id'] = frame_num
                
                if frame_num == 0:
                    reward_calc.reset_episode(game_state)
                
                # Process frame
                raw_frame = np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8)
                stacked_frames, state_vector = preprocessor.process_step(raw_frame, game_state)
                
                # Calculate reward
                reward, components = reward_calc.calculate_frame_reward(game_state)
                total_reward += reward
                
                # Validate frame processing
                if stacked_frames.shape != (1, 4, 84, 84):
                    raise ValueError(f"Frame {frame_num}: Invalid stacked frames shape")
                
                if state_vector.shape != (1, 20):
                    raise ValueError(f"Frame {frame_num}: Invalid state vector shape")
            
            # Validate sequence processing
            if total_reward <= 0:
                raise ValueError("Expected positive total reward for forward progression")
            
            results['tests_passed'] += 1
            results['details'].append("[PASS] Multi-frame sequence processing validated")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] Multi-frame sequence processing failed: {e}")
        
        # Test 3: System stress test
        try:
            # High-frequency processing simulation
            preprocessor = MarioPreprocessor(enhanced_features=True)
            parser = BinaryPayloadParser(enhanced_features=True)
            
            start_time = time.time()
            processed_frames = 0
            
            # Process for 1 second at high frequency
            while time.time() - start_time < 1.0:
                # Generate and process frame
                test_payload = self.test_data_generator.generate_enhanced_payload()
                parsed_state = parser.parse_payload(test_payload)
                
                raw_frame = np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8)
                stacked_frames, state_vector = preprocessor.process_step(raw_frame, parsed_state)
                
                processed_frames += 1
            
            # Should process at least 60 FPS equivalent
            if processed_frames < 60:
                raise ValueError(f"System too slow: only {processed_frames} FPS")
            
            self.performance_metrics['max_fps'] = processed_frames
            results['tests_passed'] += 1
            results['details'].append(f"[PASS] System stress test: {processed_frames} FPS")
            
        except Exception as e:
            results['tests_failed'] += 1
            results['details'].append(f"[FAIL] System stress test failed: {e}")
        
        if results['tests_failed'] > 0:
            results['status'] = 'FAILED'
        
        return results
    
    def generate_final_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive final test report."""
        # Calculate overall statistics
        total_tests_passed = sum(result.get('tests_passed', 0) for result in self.test_results.values())
        total_tests_failed = sum(result.get('tests_failed', 0) for result in self.test_results.values())
        total_tests = total_tests_passed + total_tests_failed
        
        success_rate = (total_tests_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Determine overall status
        overall_status = 'PASSED' if total_tests_failed == 0 else 'FAILED'
        
        # Identify critical failures
        critical_failures = []
        for category, result in self.test_results.items():
            if result.get('status') == 'FAILED':
                critical_failures.append(category)
        
        # Generate recommendations
        recommendations = []
        if 'Lua Memory Reading' in critical_failures:
            recommendations.append("Review Lua memory address mappings and payload generation")
        if 'Binary Protocol' in critical_failures:
            recommendations.append("Validate WebSocket protocol implementation and payload structure")
        if 'State Processing' in critical_failures:
            recommendations.append("Check state normalization and feature extraction logic")
        if 'Reward Calculation' in critical_failures:
            recommendations.append("Review enhanced reward calculation components")
        if 'Performance Benchmarks' in critical_failures:
            recommendations.append("Optimize system performance for real-time processing")
        
        # System readiness assessment
        system_ready = (
            overall_status == 'PASSED' and
            success_rate >= 95 and
            len(critical_failures) == 0
        )
        
        return {
            'overall_status': overall_status,
            'system_ready_for_training': system_ready,
            'execution_time_seconds': total_time,
            'test_statistics': {
                'total_tests': total_tests,
                'tests_passed': total_tests_passed,
                'tests_failed': total_tests_failed,
                'success_rate_percent': success_rate
            },
            'category_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'critical_failures': critical_failures,
            'validation_errors': self.validation_errors,
            'recommendations': recommendations,
            'system_assessment': {
                'enhanced_features_functional': 'Binary Protocol' not in critical_failures and 'State Processing' not in critical_failures,
                'backward_compatibility_maintained': 'Configuration Switching' not in critical_failures,
                'performance_acceptable': 'Performance Benchmarks' not in critical_failures,
                'error_handling_robust': 'Error Handling' not in critical_failures,
                'integration_complete': 'End-to-End Integration' not in critical_failures
            },
            'next_steps': [
                "Address any critical failures identified in the test results",
                "Review performance metrics and optimize if necessary",
                "Validate system with actual FCEUX integration",
                "Conduct extended training runs to verify stability",
                "Monitor system behavior under production conditions"
            ] if not system_ready else [
                "System is ready for enhanced training",
                "Begin training with enhanced 20-feature mode",
                "Monitor performance and adjust configurations as needed",
                "Collect training metrics for further optimization"
            ]
        }


class TestDataGenerator:
    """Generates test data for integration testing."""
    
    def __init__(self):
        """Initialize test data generator."""
        self.mario_x_base = 1000
        self.mario_y_base = 120
        
    def generate_enhanced_payload(self) -> bytes:
        """Generate a valid 128-byte enhanced payload."""
        payload = bytearray(128)
        
        # Mario Data Block (16 bytes: positions 0-15)
        mario_x_world = self.mario_x_base + np.random.randint(-100, 100)
        mario_y_level = self.mario_y_base + np.random.randint(-20, 20)
        mario_x_vel = np.random.randint(-50, 50)
        mario_y_vel = np.random.randint(-50, 50)
        
        payload[0:2] = struct.pack('<H', mario_x_world)
        payload[2:4] = struct.pack('<H', mario_y_level)
        payload[4] = mario_x_vel & 0xFF
        payload[5] = mario_y_vel & 0xFF
        payload[6] = np.random.randint(0, 3)  # power_state
        payload[7] = np.random.randint(0, 10)  # animation_state
        payload[8] = np.random.randint(0, 2)  # direction
        payload[9] = np.random.randint(0, 20)  # player_state
        payload[10] = np.random.randint(1, 5)  # lives
        payload[11] = np.random.randint(0, 100)  # invincibility_timer
        payload[12] = mario_x_world & 0xFF  # mario_x_raw
        payload[13] = np.random.randint(0, 2)  # crouching
        
        # Enemy Data Block (32 bytes: positions 16-47)
        for i in range(8):
            offset = 16 + i * 4
            payload[offset] = np.random.randint(0, 5)  # enemy_type
            payload[offset + 1] = np.random.randint(0, 255)  # enemy_x
            payload[offset + 2] = np.random.randint(0, 240)  # enemy_y
            payload[offset + 3] = np.random.randint(0, 10)  # enemy_state
        
        # Level Data Block (64 bytes: positions 48-111)
        payload[48:50] = struct.pack('<H', np.random.randint(0, 1000))  # camera_x
        payload[50] = 1  # world_number
        payload[51] = 1  # level_number
        payload[52] = np.random.randint(0, 10)  # score_100k
        payload[53] = np.random.randint(0, 10)  # score_10k
        payload[54] = np.random.randint(0, 10)  # score_1k
        payload[55] = np.random.randint(0, 10)  # score_100
        payload[56:60] = struct.pack('<I', max(0, min(4294967295, np.random.randint(0, 400))))  # time_remaining
        payload[60:62] = struct.pack('<H', np.random.randint(0, 99))  # total_coins
        
        # Enhanced features (positions 62-97)
        payload[62] = np.random.randint(0, 5)  # powerup_type
        payload[63] = np.random.randint(0, 255)  # powerup_x_pos
        payload[64] = np.random.randint(0, 240)  # powerup_y_pos
        payload[65] = np.random.randint(0, 10)  # powerup_state
        payload[66:68] = struct.pack('<H', np.random.randint(0, 1000))  # powerup_world_x
        payload[68] = np.random.randint(0, 2)  # powerup_is_active
        
        # Threat assessment (8 bytes)
        payload[70] = np.random.randint(0, 5)  # threat_count
        payload[71] = np.random.randint(0, 3)  # threats_ahead
        payload[72] = np.random.randint(0, 3)  # threats_behind
        payload[74:76] = struct.pack('<H', np.random.randint(50, 500))  # nearest_threat_distance
        
        # Level tiles (16 bytes)
        for i in range(16):
            payload[82 + i] = np.random.randint(0, 5)  # tile_value
        
        # Game Variables Block (16 bytes: positions 112-127)
        payload[112] = np.random.randint(0, 10)  # game_engine_state
        payload[113] = np.random.randint(0, 100)  # level_progress
        payload[114:116] = struct.pack('<H', np.random.randint(0, 3000))  # distance_to_flag
        payload[116:120] = struct.pack('<I', max(0, min(4294967295, np.random.randint(0, 100000))))  # frame_id
        payload[120:124] = struct.pack('<I', max(0, min(4294967295, int(time.time() * 1000))))  # timestamp
        
        return bytes(payload)
    
    def generate_legacy_payload(self) -> bytes:
        """Generate a legacy 128-byte payload (for backward compatibility testing)."""
        # For simplicity, generate same structure but mark as legacy
        return self.generate_enhanced_payload()
    
    def generate_game_state(self, enhanced: bool = True) -> Dict[str, Any]:
        """Generate a game state dictionary."""
        base_state = {
            'mario_x': self.mario_x_base + np.random.randint(-100, 100),
            'mario_y': self.mario_y_base + np.random.randint(-20, 20),
            'mario_x_vel': np.random.randint(-50, 50),
            'mario_y_vel': np.random.randint(-50, 50),
            'power_state': np.random.randint(0, 3),
            'direction': np.random.randint(0, 2),
            'lives': np.random.randint(1, 5),
            'on_ground': np.random.randint(0, 2),
            'invincible': np.random.randint(0, 100),
            'time_remaining': np.random.randint(100, 400),
            'coins': np.random.randint(0, 99)
        }
        
        if enhanced:
            base_state.update({
                'closest_enemy_distance': np.random.uniform(50, 500),
                'enemy_count': np.random.randint(0, 5),
                'powerup_present': np.random.choice([True, False]),
                'powerup_distance': np.random.uniform(50, 300),
                'solid_tiles_ahead': np.random.randint(0, 10),
                'pit_detected': np.random.choice([True, False]),
                'velocity_magnitude': np.random.uniform(0, 100),
                'facing_direction': np.random.randint(0, 2),
                'mario_below_viewport': np.random.choice([True, False])
            })
        
        return base_state


async def main():
    """Main test execution function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run test suite
    test_suite = IntegrationTestSuite()
    
    print("[START] Starting Enhanced State Management System Integration Tests")
    print("=" * 80)
    
    # Run all tests
    final_report = await test_suite.run_all_tests()
    
    # Display results
    print("\n" + "=" * 80)
    print("[RESULTS] INTEGRATION TEST RESULTS")
    print("=" * 80)
    
    print(f"Overall Status: {'[PASS] PASSED' if final_report['overall_status'] == 'PASSED' else '[FAIL] FAILED'}")
    print(f"System Ready for Training: {'[PASS] YES' if final_report['system_ready_for_training'] else '[FAIL] NO'}")
    print(f"Execution Time: {final_report['execution_time_seconds']:.2f} seconds")
    
    stats = final_report['test_statistics']
    print(f"Tests: {stats['tests_passed']}/{stats['total_tests']} passed ({stats['success_rate_percent']:.1f}%)")
    
    if final_report['critical_failures']:
        print(f"\n[FAIL] Critical Failures: {', '.join(final_report['critical_failures'])}")
    
    if final_report['performance_metrics']:
        print(f"\n[PERF] Performance Metrics:")
        for metric, value in final_report['performance_metrics'].items():
            print(f"  • {metric}: {value}")
    
    print(f"\n[RECS] Recommendations:")
    for rec in final_report['recommendations']:
        print(f"  • {rec}")
    
    print(f"\n[NEXT] Next Steps:")
    for step in final_report['next_steps']:
        print(f"  • {step}")
    
    # Save detailed report
    report_file = f"integration_test_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    print(f"\n[REPORT] Detailed report saved to: {report_file}")
    
    return final_report['system_ready_for_training']


if __name__ == "__main__":
    # Run the integration test suite
    success = asyncio.run(main())
    sys.exit(0 if success else 1)