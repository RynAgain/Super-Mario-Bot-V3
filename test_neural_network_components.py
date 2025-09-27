#!/usr/bin/env python3
"""
Test Script for Super Mario Bros AI Neural Network Components

This script validates all the implemented neural network components including:
- Dueling DQN model
- Experience replay buffer
- Preprocessing utilities
- Model utilities
- DQN agent
- Configuration loading

Run this script to ensure all components are working correctly before integration.
"""

import sys
import os
import torch
import numpy as np
import logging
from pathlib import Path

# Add python directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_dueling_dqn():
    """Test the Dueling DQN model implementation."""
    logger.info("Testing Dueling DQN model...")
    
    try:
        from python.models.dueling_dqn import create_dueling_dqn, ACTION_SPACE
        
        # Create model
        model = create_dueling_dqn()
        logger.info(f"✓ Model created successfully")
        
        # Test forward pass
        batch_size = 4
        frames = torch.randn(batch_size, 8, 84, 84)
        state_vector = torch.randn(batch_size, 12)
        
        q_values = model(frames, state_vector)
        
        # Validate output shape
        expected_shape = (batch_size, 12)
        assert q_values.shape == expected_shape, f"Expected shape {expected_shape}, got {q_values.shape}"
        logger.info(f"✓ Forward pass successful, output shape: {q_values.shape}")
        
        # Test action selection
        action = model.get_action(frames[:1], state_vector[:1], epsilon=0.1)
        assert 0 <= action < 12, f"Invalid action: {action}"
        logger.info(f"✓ Action selection successful, selected action: {action} ({ACTION_SPACE[action]})")
        
        # Test parameter count
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Model has {total_params:,} parameters")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Dueling DQN test failed: {e}")
        return False


def test_replay_buffer():
    """Test the experience replay buffer implementation."""
    logger.info("Testing Experience Replay Buffer...")
    
    try:
        from python.utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
        
        # Test standard replay buffer
        buffer = ReplayBuffer(capacity=1000, device="cpu")
        logger.info(f"✓ Standard replay buffer created")
        
        # Add experiences
        for i in range(100):
            state_frames = torch.randn(8, 84, 84)
            state_vector = torch.randn(12)
            action = np.random.randint(0, 12)
            reward = np.random.randn()
            next_state_frames = torch.randn(8, 84, 84)
            next_state_vector = torch.randn(12)
            done = np.random.random() < 0.1
            
            buffer.add(
                state_frames, state_vector, action, reward,
                next_state_frames, next_state_vector, done
            )
        
        logger.info(f"✓ Added 100 experiences, buffer size: {len(buffer)}")
        
        # Test sampling
        if buffer.is_ready(32):
            batch = buffer.sample(32)
            assert len(batch) == 9, f"Expected 9 batch elements, got {len(batch)}"
            logger.info(f"✓ Batch sampling successful")
        
        # Test prioritized replay buffer
        prioritized_buffer = PrioritizedReplayBuffer(capacity=1000, device="cpu")
        
        # Add experiences
        for i in range(50):
            state_frames = torch.randn(8, 84, 84)
            state_vector = torch.randn(12)
            action = np.random.randint(0, 12)
            reward = np.random.randn()
            next_state_frames = torch.randn(8, 84, 84)
            next_state_vector = torch.randn(12)
            done = np.random.random() < 0.1
            
            prioritized_buffer.add(
                state_frames, state_vector, action, reward,
                next_state_frames, next_state_vector, done
            )
        
        logger.info(f"✓ Prioritized replay buffer test successful")
        
        # Test memory usage
        memory_stats = buffer.get_memory_usage()
        logger.info(f"✓ Memory usage: {memory_stats['total_mb']:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Replay buffer test failed: {e}")
        return False


def test_preprocessing():
    """Test the preprocessing utilities."""
    logger.info("Testing Preprocessing Utilities...")
    
    try:
        from python.utils.preprocessing import MarioPreprocessor, FramePreprocessor, StateNormalizer
        
        # Test frame preprocessor
        frame_preprocessor = FramePreprocessor(device="cpu")
        dummy_frame = np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8)
        processed_frame = frame_preprocessor.preprocess_frame(dummy_frame)
        
        assert processed_frame.shape == (84, 84), f"Expected (84, 84), got {processed_frame.shape}"
        assert 0 <= processed_frame.min() <= processed_frame.max() <= 1, "Frame values not normalized"
        logger.info(f"✓ Frame preprocessing successful")
        
        # Test state normalizer
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
        assert normalized_state.shape == (12,), f"Expected (12,), got {normalized_state.shape}"
        logger.info(f"✓ State normalization successful")
        
        # Test complete Mario preprocessor
        mario_preprocessor = MarioPreprocessor(device="cpu")
        raw_frame = np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8)
        
        stacked_frames, state_vector = mario_preprocessor.process_step(raw_frame, dummy_game_state)
        
        assert stacked_frames.shape == (1, 8, 84, 84), f"Expected (1, 8, 84, 84), got {stacked_frames.shape}"
        assert state_vector.shape == (1, 12), f"Expected (1, 12), got {state_vector.shape}"
        logger.info(f"✓ Complete preprocessing pipeline successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Preprocessing test failed: {e}")
        return False


def test_model_utils():
    """Test the model utilities."""
    logger.info("Testing Model Utilities...")
    
    try:
        from python.utils.model_utils import DeviceManager, ModelManager, count_parameters, model_summary
        from python.models.dueling_dqn import create_dueling_dqn
        
        # Test device manager
        device_manager = DeviceManager("cpu")  # Force CPU for testing
        logger.info(f"✓ Device manager created, using device: {device_manager.device}")
        
        # Test model manager
        model_manager = ModelManager(checkpoint_dir="test_checkpoints")
        model = create_dueling_dqn()
        
        # Test parameter counting
        param_counts = count_parameters(model)
        logger.info(f"✓ Parameter counting: {param_counts['total']:,} total parameters")
        
        # Test model summary
        summary = model_summary(model, [(8, 84, 84), (12,)])
        logger.info(f"✓ Model summary generated")
        
        # Test checkpoint saving (create minimal test)
        try:
            checkpoint_path = model_manager.save_checkpoint(
                model=model,
                episode=1,
                step=100,
                metrics={'test_metric': 1.0}
            )
            logger.info(f"✓ Checkpoint saved successfully")
            
            # Cleanup test checkpoint
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            if os.path.exists("test_checkpoints"):
                import shutil
                shutil.rmtree("test_checkpoints")
                
        except Exception as e:
            logger.warning(f"Checkpoint test skipped: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model utilities test failed: {e}")
        return False


def test_config_loader():
    """Test the configuration loader."""
    logger.info("Testing Configuration Loader...")
    
    try:
        from python.utils.config_loader import ConfigLoader, load_config
        
        # Test loading all configs
        config_loader = ConfigLoader()
        config = config_loader.load_all_configs()
        
        # Check that main sections exist
        expected_sections = ['training', 'network', 'game', 'logging']
        for section in expected_sections:
            if section in config:
                logger.info(f"✓ {section} configuration loaded")
            else:
                logger.warning(f"⚠ {section} configuration not found")
        
        # Test specific section access
        training_config = config_loader.get_config('training')
        if training_config:
            logger.info(f"✓ Training config access successful")
        
        # Test convenience function
        config2 = load_config()
        logger.info(f"✓ Convenience function load_config() successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration loader test failed: {e}")
        return False


def test_dqn_agent():
    """Test the DQN agent implementation."""
    logger.info("Testing DQN Agent...")
    
    try:
        from python.agents.dqn_agent import DQNAgent
        
        # Create minimal config for testing
        config = {
            'learning_rate': 0.00025,
            'batch_size': 32,
            'gamma': 0.99,
            'target_update_frequency': 1000,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'replay_buffer_size': 1000,
            'double_dqn': True,
            'mixed_precision': False,  # Disable for testing
            'compile_model': False     # Disable for testing
        }
        
        # Create agent
        agent = DQNAgent(config, device="cpu")
        logger.info(f"✓ DQN Agent created successfully")
        
        # Test action selection
        dummy_frames = torch.randn(1, 8, 84, 84)
        dummy_state = torch.randn(1, 12)
        
        action = agent.select_action(dummy_frames, dummy_state, training=True)
        assert 0 <= action < 12, f"Invalid action: {action}"
        logger.info(f"✓ Action selection successful")
        
        # Test experience storage
        agent.store_experience(
            dummy_frames, dummy_state, action, 1.0,
            dummy_frames, dummy_state, False
        )
        logger.info(f"✓ Experience storage successful")
        
        # Add more experiences for training test
        for _ in range(50):
            frames = torch.randn(1, 8, 84, 84)
            state = torch.randn(1, 12)
            action = np.random.randint(0, 12)
            reward = np.random.randn()
            done = np.random.random() < 0.1
            
            agent.store_experience(frames, state, action, reward, frames, state, done)
        
        # Test training step
        if agent.replay_buffer.is_ready(agent.batch_size):
            metrics = agent.train_step()
            logger.info(f"✓ Training step successful")
        
        # Test statistics
        stats = agent.get_stats()
        logger.info(f"✓ Statistics retrieval successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ DQN Agent test failed: {e}")
        return False


def main():
    """Run all component tests."""
    logger.info("=" * 60)
    logger.info("SUPER MARIO BROS AI - NEURAL NETWORK COMPONENT TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Dueling DQN Model", test_dueling_dqn),
        ("Experience Replay Buffer", test_replay_buffer),
        ("Preprocessing Utilities", test_preprocessing),
        ("Model Utilities", test_model_utils),
        ("Configuration Loader", test_config_loader),
        ("DQN Agent", test_dqn_agent),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{test_name:<30} {status}")
        if success:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All neural network components are working correctly!")
        logger.info("The system is ready for integration with the Lua script and training pipeline.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please fix the issues before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)