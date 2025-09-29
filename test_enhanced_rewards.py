#!/usr/bin/env python3
"""
Test script for the enhanced reward calculation system.

This script tests the enhanced reward calculator with both 12-feature and 20-feature modes
to ensure backward compatibility and proper functionality of all new reward components.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from python.environment.reward_calculator import RewardCalculator, RewardComponents
import yaml

def load_test_config():
    """Load test configuration with enhanced rewards enabled."""
    config = {
        'enhanced_rewards': {
            'enabled': True,
            'powerup_collection_reward': 50.0,
            'enemy_elimination_reward': 25.0,
            'environmental_navigation_reward': 5.0,
            'velocity_movement_multiplier': 0.1,
            'strategic_positioning_reward': 2.0,
            'safe_distance_threshold': 100.0,
            'pit_avoidance_reward': 10.0,
            'obstacle_navigation_reward': 5.0,
            'forward_momentum_threshold': 10.0,
            'pit_death_penalty': -100.0,
            'enemy_collision_penalty': -50.0,
            'time_death_penalty': -25.0,
            'general_death_penalty': -10.0,
            'feature_weights': {
                'powerup_collection': 1.0,
                'enemy_elimination': 1.0,
                'environmental_awareness': 1.0,
                'velocity_movement': 1.0,
                'strategic_positioning': 1.0
            }
        }
    }
    return config

def create_test_state_12_feature():
    """Create a test game state for 12-feature mode."""
    return {
        'mario_x': 1000,
        'mario_y': 120,
        'mario_x_vel': 20,
        'mario_y_vel': 0,
        'power_state': 1,
        'on_ground': 1,
        'direction': 1,
        'lives': 3,
        'invincible': 0,
        'time_remaining': 350,
        'level_progress': 0.3
    }

def create_test_state_20_feature():
    """Create a test game state for 20-feature mode with enhanced features."""
    base_state = create_test_state_12_feature()
    base_state.update({
        # Enhanced features
        'closest_enemy_distance': 150.0,
        'enemy_count': 2,
        'powerup_present': True,
        'powerup_distance': 80.0,
        'solid_tiles_ahead': 3,
        'pit_detected': False,
        'velocity_magnitude': 25.0,
        'facing_direction': 1,
        'mario_below_viewport': False
    })
    return base_state

def test_basic_reward_calculation():
    """Test basic reward calculation (12-feature mode)."""
    print("Testing basic reward calculation (12-feature mode)...")
    
    config = load_test_config()
    calculator = RewardCalculator(config, enhanced_features=False)
    
    # Initial state
    initial_state = create_test_state_12_feature()
    calculator.reset_episode(initial_state)
    
    # Test forward movement reward
    next_state = initial_state.copy()
    next_state['mario_x'] = 1050  # Move forward 50 pixels
    
    reward, components = calculator.calculate_frame_reward(next_state)
    
    print(f"Forward movement reward: {reward:.2f}")
    print(f"Components: {components.to_dict()}")
    
    assert reward > 0, "Forward movement should give positive reward"
    assert components.distance_reward > 0, "Distance reward should be positive"
    
    print("✓ Basic reward calculation test passed\n")

def test_enhanced_powerup_collection():
    """Test power-up collection reward."""
    print("Testing power-up collection reward...")
    
    config = load_test_config()
    calculator = RewardCalculator(config, enhanced_features=True)
    
    # Initial state
    initial_state = create_test_state_20_feature()
    initial_state['power_state'] = 0  # Small Mario
    calculator.reset_episode(initial_state)
    
    # Power-up collected
    next_state = initial_state.copy()
    next_state['power_state'] = 1  # Big Mario
    next_state['mario_x'] = 1010  # Small forward movement
    
    reward, components = calculator.calculate_frame_reward(next_state)
    
    print(f"Power-up collection reward: {reward:.2f}")
    print(f"Power-up component: {components.powerup_collection_reward:.2f}")
    
    assert components.powerup_collection_reward == 50.0, "Power-up collection should give 50 points"
    
    print("✓ Power-up collection test passed\n")

def test_enhanced_enemy_elimination():
    """Test enemy elimination reward."""
    print("Testing enemy elimination reward...")
    
    config = load_test_config()
    calculator = RewardCalculator(config, enhanced_features=True)
    
    # Initial state with 3 enemies
    initial_state = create_test_state_20_feature()
    initial_state['enemy_count'] = 3
    calculator.reset_episode(initial_state)
    
    # Enemy eliminated
    next_state = initial_state.copy()
    next_state['enemy_count'] = 2  # One enemy eliminated
    next_state['mario_x'] = 1010  # Small forward movement
    
    reward, components = calculator.calculate_frame_reward(next_state)
    
    print(f"Enemy elimination reward: {reward:.2f}")
    print(f"Enemy elimination component: {components.enemy_elimination_reward:.2f}")
    
    assert components.enemy_elimination_reward == 25.0, "Enemy elimination should give 25 points"
    
    print("✓ Enemy elimination test passed\n")

def test_enhanced_environmental_navigation():
    """Test environmental navigation rewards."""
    print("Testing environmental navigation rewards...")
    
    config = load_test_config()
    calculator = RewardCalculator(config, enhanced_features=True)
    
    # Initial state
    initial_state = create_test_state_20_feature()
    calculator.reset_episode(initial_state)
    
    # Pit detected but moving forward (avoidance)
    next_state = initial_state.copy()
    next_state['pit_detected'] = True
    next_state['mario_x_vel'] = 15  # Moving forward
    next_state['mario_x'] = 1010  # Small forward movement
    
    reward, components = calculator.calculate_frame_reward(next_state)
    
    print(f"Pit avoidance reward: {reward:.2f}")
    print(f"Environmental component: {components.environmental_navigation_reward:.2f}")
    
    assert components.environmental_navigation_reward == 10.0, "Pit avoidance should give 10 points"
    
    print("✓ Environmental navigation test passed\n")

def test_enhanced_velocity_movement():
    """Test velocity-based movement rewards."""
    print("Testing velocity-based movement rewards...")
    
    config = load_test_config()
    calculator = RewardCalculator(config, enhanced_features=True)
    
    # Initial state
    initial_state = create_test_state_20_feature()
    calculator.reset_episode(initial_state)
    
    # High forward velocity
    next_state = initial_state.copy()
    next_state['mario_x_vel'] = 25  # High forward velocity
    next_state['velocity_magnitude'] = 25.0
    next_state['mario_x'] = 1010  # Small forward movement
    
    reward, components = calculator.calculate_frame_reward(next_state)
    
    print(f"Velocity movement reward: {reward:.2f}")
    print(f"Velocity component: {components.velocity_movement_reward:.2f}")
    
    expected_velocity_reward = 25.0 * 0.1  # velocity_magnitude * multiplier
    assert abs(components.velocity_movement_reward - expected_velocity_reward) < 0.01, \
        f"Velocity reward should be {expected_velocity_reward}"
    
    print("✓ Velocity movement test passed\n")

def test_enhanced_strategic_positioning():
    """Test strategic positioning rewards."""
    print("Testing strategic positioning rewards...")
    
    config = load_test_config()
    calculator = RewardCalculator(config, enhanced_features=True)
    
    # Initial state
    initial_state = create_test_state_20_feature()
    calculator.reset_episode(initial_state)
    
    # Safe distance from enemies
    next_state = initial_state.copy()
    next_state['closest_enemy_distance'] = 150.0  # Safe distance (> 100)
    next_state['mario_x'] = 1010  # Small forward movement
    
    reward, components = calculator.calculate_frame_reward(next_state)
    
    print(f"Strategic positioning reward: {reward:.2f}")
    print(f"Strategic component: {components.strategic_positioning_reward:.2f}")
    
    assert components.strategic_positioning_reward == 2.0, "Strategic positioning should give 2 points"
    
    print("✓ Strategic positioning test passed\n")

def test_enhanced_death_penalties():
    """Test enhanced death penalty system."""
    print("Testing enhanced death penalty system...")
    
    config = load_test_config()
    calculator = RewardCalculator(config, enhanced_features=True)
    
    # Initial state
    initial_state = create_test_state_20_feature()
    calculator.reset_episode(initial_state)
    
    # Pit death
    death_state = initial_state.copy()
    death_state['lives'] = 2  # Lost a life
    death_state['mario_below_viewport'] = True  # Fell into pit
    
    reward, components = calculator.calculate_frame_reward(death_state)
    
    print(f"Pit death penalty: {reward:.2f}")
    print(f"Enhanced death penalty: {components.enhanced_death_penalty:.2f}")
    
    assert components.enhanced_death_penalty == -100.0, "Pit death should give -100 penalty"
    
    print("✓ Enhanced death penalty test passed\n")

def test_backward_compatibility():
    """Test backward compatibility with 12-feature mode."""
    print("Testing backward compatibility (12-feature mode)...")
    
    config = load_test_config()
    calculator_12 = RewardCalculator(config, enhanced_features=False)
    calculator_20 = RewardCalculator(config, enhanced_features=True)
    
    # Test with 12-feature state
    state_12 = create_test_state_12_feature()
    
    calculator_12.reset_episode(state_12)
    calculator_20.reset_episode(state_12)
    
    # Forward movement
    next_state = state_12.copy()
    next_state['mario_x'] = 1050
    
    reward_12, components_12 = calculator_12.calculate_frame_reward(next_state)
    reward_20, components_20 = calculator_20.calculate_frame_reward(next_state)
    
    print(f"12-feature reward: {reward_12:.2f}")
    print(f"20-feature reward (with 12-feature state): {reward_20:.2f}")
    
    # Basic rewards should be similar (enhanced features won't trigger without enhanced state)
    assert abs(components_12.distance_reward - components_20.distance_reward) < 0.01, \
        "Distance rewards should be similar in both modes"
    
    print("✓ Backward compatibility test passed\n")

def test_reward_component_logging():
    """Test reward component logging and statistics."""
    print("Testing reward component logging...")
    
    config = load_test_config()
    calculator = RewardCalculator(config, enhanced_features=True)
    
    # Initial state with small Mario
    initial_state = create_test_state_20_feature()
    initial_state['power_state'] = 0  # Start as small Mario
    initial_state['enemy_count'] = 3  # Start with 3 enemies
    calculator.reset_episode(initial_state)
    
    # Simulate several reward events
    states = [
        # Power-up collection (small -> big)
        {**initial_state, 'power_state': 1, 'mario_x': 1010},
        # Enemy elimination (3 -> 2 enemies)
        {**initial_state, 'power_state': 1, 'enemy_count': 2, 'mario_x': 1020},
        # Pit avoidance
        {**initial_state, 'power_state': 1, 'enemy_count': 2, 'pit_detected': True, 'mario_x_vel': 15, 'mario_x': 1030}
    ]
    
    total_rewards = []
    for state in states:
        reward, components = calculator.calculate_frame_reward(state)
        total_rewards.append(reward)
        print(f"State reward: {reward:.2f}, Components: {components.to_dict()}")
    
    # Check enhanced statistics
    enhanced_stats = calculator.get_enhanced_reward_stats()
    print(f"Enhanced stats: {enhanced_stats}")
    
    assert enhanced_stats['powerup_collections'] >= 1, "Should track power-up collections"
    assert enhanced_stats['enemy_eliminations'] >= 1, "Should track enemy eliminations"
    assert enhanced_stats['pit_avoidances'] >= 1, "Should track pit avoidances"
    
    print("✓ Reward component logging test passed\n")

def run_all_tests():
    """Run all enhanced reward system tests."""
    print("=" * 60)
    print("ENHANCED REWARD SYSTEM TESTS")
    print("=" * 60)
    
    try:
        test_basic_reward_calculation()
        test_enhanced_powerup_collection()
        test_enhanced_enemy_elimination()
        test_enhanced_environmental_navigation()
        test_enhanced_velocity_movement()
        test_enhanced_strategic_positioning()
        test_enhanced_death_penalties()
        test_backward_compatibility()
        test_reward_component_logging()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("Enhanced reward system is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        raise

if __name__ == "__main__":
    run_all_tests()