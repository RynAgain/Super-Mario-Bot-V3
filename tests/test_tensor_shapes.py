#!/usr/bin/env python3
"""
Test script to validate tensor shape consistency across the Super Mario Bot system.
This script tests that all components work together with 4-frame stacks.
"""

import torch
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from python.models.dueling_dqn import create_dueling_dqn
from python.utils.replay_buffer import ReplayBuffer
from python.agents.dqn_agent import DQNAgent

def test_tensor_shapes():
    """Test tensor shape consistency across all components."""
    print("Testing tensor shape consistency...")
    
    # Configuration
    frame_stack_size = 4
    frame_size = (84, 84)
    state_vector_size = 12
    num_actions = 12
    batch_size = 32
    
    print(f"Configuration:")
    print(f"  - Frame stack size: {frame_stack_size}")
    print(f"  - Frame size: {frame_size}")
    print(f"  - State vector size: {state_vector_size}")
    print(f"  - Number of actions: {num_actions}")
    print(f"  - Batch size: {batch_size}")
    print()
    
    # Test 1: Model creation and forward pass
    print("Test 1: Model creation and forward pass")
    try:
        model = create_dueling_dqn()
        print(f"✓ Model created successfully")
        print(f"  - Expected input channels: {model.frame_stack_size}")
        
        # Test forward pass
        test_frames = torch.randn(1, frame_stack_size, *frame_size)
        test_state = torch.randn(1, state_vector_size)
        
        q_values = model(test_frames, test_state)
        print(f"✓ Forward pass successful")
        print(f"  - Input frames shape: {test_frames.shape}")
        print(f"  - Input state shape: {test_state.shape}")
        print(f"  - Output Q-values shape: {q_values.shape}")
        
        if q_values.shape == (1, num_actions):
            print("✓ Output shape is correct")
        else:
            print(f"✗ Output shape mismatch: expected (1, {num_actions}), got {q_values.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False
    
    print()
    
    # Test 2: Replay buffer
    print("Test 2: Replay buffer")
    try:
        buffer = ReplayBuffer(
            capacity=1000,
            frame_stack_size=frame_stack_size,
            frame_size=frame_size,
            state_vector_size=state_vector_size,
            device="cpu"
        )
        print(f"✓ Replay buffer created successfully")
        
        # Add some experiences
        for i in range(100):
            state_frames = torch.randn(frame_stack_size, *frame_size)
            state_vector = torch.randn(state_vector_size)
            action = np.random.randint(0, num_actions)
            reward = np.random.randn()
            next_state_frames = torch.randn(frame_stack_size, *frame_size)
            next_state_vector = torch.randn(state_vector_size)
            done = np.random.random() < 0.1
            
            buffer.add(
                state_frames, state_vector, action, reward,
                next_state_frames, next_state_vector, done
            )
        
        print(f"✓ Added 100 experiences to buffer")
        
        # Test sampling
        if buffer.is_ready(batch_size):
            batch = buffer.sample(batch_size)
            print(f"✓ Sampling successful")
            print(f"  - State frames shape: {batch[0].shape}")
            print(f"  - State vectors shape: {batch[1].shape}")
            print(f"  - Next state frames shape: {batch[4].shape}")
            print(f"  - Next state vectors shape: {batch[5].shape}")
            
            # Validate shapes
            expected_frame_shape = (batch_size, frame_stack_size, *frame_size)
            expected_vector_shape = (batch_size, state_vector_size)
            
            if batch[0].shape == expected_frame_shape and batch[1].shape == expected_vector_shape:
                print("✓ Batch shapes are correct")
            else:
                print(f"✗ Batch shape mismatch")
                print(f"  Expected frames: {expected_frame_shape}, got: {batch[0].shape}")
                print(f"  Expected vectors: {expected_vector_shape}, got: {batch[1].shape}")
                return False
        else:
            print("✗ Buffer not ready for sampling")
            return False
            
    except Exception as e:
        print(f"✗ Replay buffer test failed: {e}")
        return False
    
    print()
    
    # Test 3: DQN Agent
    print("Test 3: DQN Agent")
    try:
        config = {
            'learning_rate': 0.00025,
            'batch_size': 32,
            'gamma': 0.99,
            'target_update_frequency': 1000,  # Now in steps
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.9995,  # Slower decay
            'replay_buffer_size': 10000,
            'double_dqn': True,
            'mixed_precision': False,  # Disable for testing
            'compile_model': False
        }
        
        agent = DQNAgent(config, device="cpu")
        print(f"✓ DQN Agent created successfully")
        
        # Test action selection
        test_frames = torch.randn(1, frame_stack_size, *frame_size)
        test_state = torch.randn(1, state_vector_size)
        
        action = agent.select_action(test_frames, test_state, training=True)
        print(f"✓ Action selection successful: action {action}")
        
        # Test experience storage
        agent.store_experience(
            test_frames, test_state, action, 1.0,
            test_frames, test_state, False
        )
        print(f"✓ Experience storage successful")
        
        # Add more experiences for training test
        for i in range(100):
            frames = torch.randn(1, frame_stack_size, *frame_size)
            state = torch.randn(1, state_vector_size)
            action = np.random.randint(0, num_actions)
            reward = np.random.randn()
            done = np.random.random() < 0.1
            
            agent.store_experience(frames, state, action, reward, frames, state, done)
        
        # Test training step
        if agent.replay_buffer.is_ready(agent.batch_size):
            metrics = agent.train_step()
            print(f"✓ Training step successful")
            print(f"  - Loss: {metrics.get('loss', 'N/A')}")
            print(f"  - Mean Q-value: {metrics.get('mean_q_value', 'N/A')}")
        else:
            print("✗ Agent replay buffer not ready")
            return False
            
    except Exception as e:
        print(f"✗ DQN Agent test failed: {e}")
        return False
    
    print()
    
    # Test 4: End-to-end compatibility
    print("Test 4: End-to-end compatibility")
    try:
        # Test that model can process replay buffer samples
        batch = buffer.sample(batch_size)
        state_frames, state_vectors = batch[0], batch[1]
        
        # Process through model
        with torch.no_grad():
            q_values = model(state_frames, state_vectors)
        
        print(f"✓ End-to-end processing successful")
        print(f"  - Batch frames shape: {state_frames.shape}")
        print(f"  - Batch vectors shape: {state_vectors.shape}")
        print(f"  - Output Q-values shape: {q_values.shape}")
        
        if q_values.shape == (batch_size, num_actions):
            print("✓ End-to-end shapes are correct")
        else:
            print(f"✗ End-to-end shape mismatch: expected ({batch_size}, {num_actions}), got {q_values.shape}")
            return False
            
    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")
        return False
    
    print()
    print("🎉 All tensor shape tests passed!")
    print("The system is now consistent with 4-frame stacks.")
    return True

if __name__ == "__main__":
    success = test_tensor_shapes()
    sys.exit(0 if success else 1)