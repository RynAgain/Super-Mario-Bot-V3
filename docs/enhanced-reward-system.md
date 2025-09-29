# Enhanced Reward System Documentation

## Overview

The enhanced reward calculation system has been implemented to utilize the comprehensive 20-feature state information from the upgraded memory reading and state processing systems. This system provides more nuanced and strategic reward signals to improve AI training effectiveness.

## Key Features

### 1. Enhanced Reward Components

The system now includes six new reward components in addition to the existing basic rewards:

- **Power-up Collection Rewards** (+50 points): Rewards Mario for collecting power-ups (mushrooms, fire flowers)
- **Enemy Elimination Bonuses** (+25 points per enemy): Rewards Mario for defeating enemies
- **Environmental Navigation Bonuses** (+5-10 points): Rewards for avoiding pits and navigating obstacles
- **Velocity-based Movement Rewards** (variable): Encourages forward momentum and speed
- **Strategic Positioning Rewards** (+2 points): Rewards maintaining safe distance from enemies
- **Enhanced Death Penalties** (-10 to -100 points): Specific penalties based on death cause

### 2. Enhanced Death Penalty System

The system now differentiates between death causes and applies appropriate penalties:

- **Pit Death**: -100 points (most severe)
- **Enemy Collision**: -50 points
- **Timeout Death**: -25 points
- **General Death**: -10 points (fallback)

### 3. Backward Compatibility

The system maintains full backward compatibility with the existing 12-feature mode:
- When `enhanced_features=False`, only basic rewards are calculated
- When `enhanced_features=True`, both basic and enhanced rewards are calculated
- Configuration allows selective enabling/disabling of enhanced features

## Configuration

### Training Configuration (config/training_config.yaml)

```yaml
# Enhanced Reward System Configuration
enhanced_rewards:
  # Master switch for enhanced rewards
  enabled: false                   # Enable enhanced reward calculation
  
  # Reward values for enhanced features
  powerup_collection_reward: 50.0  # Reward for collecting power-ups
  enemy_elimination_reward: 25.0   # Reward per enemy eliminated
  environmental_navigation_reward: 5.0  # Base reward for environmental navigation
  velocity_movement_multiplier: 0.1     # Multiplier for velocity-based rewards
  strategic_positioning_reward: 2.0     # Reward for safe positioning
  
  # Environmental navigation settings
  pit_avoidance_reward: 10.0       # Reward for avoiding pits
  obstacle_navigation_reward: 5.0  # Reward for navigating obstacles
  forward_momentum_threshold: 10.0 # Minimum velocity for momentum reward
  safe_distance_threshold: 100.0   # Safe distance from enemies (pixels)
  
  # Enhanced death penalties
  pit_death_penalty: -100.0        # Penalty for falling into pits
  enemy_collision_penalty: -50.0   # Penalty for enemy collisions
  time_death_penalty: -25.0        # Penalty for timeout deaths
  general_death_penalty: -10.0     # General death penalty
  
  # Feature weights for enhanced rewards
  feature_weights:
    powerup_collection: 1.0        # Weight for power-up collection rewards
    enemy_elimination: 1.0         # Weight for enemy elimination rewards
    environmental_awareness: 1.0   # Weight for environmental navigation
    velocity_movement: 1.0         # Weight for velocity-based rewards
    strategic_positioning: 1.0     # Weight for strategic positioning
```

### Network Configuration

Ensure the network is configured for enhanced features:

```yaml
network:
  state_vector_size: 20           # Use 20-feature mode
  enhanced_features: true         # Enable enhanced features
```

## Usage

### Initializing the Enhanced Reward Calculator

```python
from python.environment.reward_calculator import RewardCalculator

# Load configuration
config = load_config('config/training_config.yaml')

# Initialize with enhanced features
reward_calculator = RewardCalculator(
    config=config,
    enhanced_features=True
)
```

### Calculating Rewards

```python
# Reset for new episode
initial_state = get_initial_game_state()
reward_calculator.reset_episode(initial_state)

# Calculate frame reward
current_state = get_current_game_state()
total_reward, components = reward_calculator.calculate_frame_reward(current_state)

# Access individual components
print(f"Power-up reward: {components.powerup_collection_reward}")
print(f"Enemy elimination reward: {components.enemy_elimination_reward}")
print(f"Environmental reward: {components.environmental_navigation_reward}")
print(f"Velocity reward: {components.velocity_movement_reward}")
print(f"Strategic reward: {components.strategic_positioning_reward}")
print(f"Enhanced death penalty: {components.enhanced_death_penalty}")
```

### Monitoring Enhanced Statistics

```python
# Get enhanced reward statistics
enhanced_stats = reward_calculator.get_enhanced_reward_stats()

print(f"Power-ups collected: {enhanced_stats['powerup_collections']}")
print(f"Enemies eliminated: {enhanced_stats['enemy_eliminations']}")
print(f"Pits avoided: {enhanced_stats['pit_avoidances']}")
print(f"Safe positioning ratio: {enhanced_stats['safe_positioning_ratio']:.2f}")
```

## State Requirements

The enhanced reward system requires the following state information:

### Basic State (12-feature mode)
- `mario_x`, `mario_y`: Mario's position
- `mario_x_vel`, `mario_y_vel`: Mario's velocity
- `power_state`: Mario's power level (0=small, 1=big, 2=fire)
- `lives`: Number of lives remaining
- `on_ground`: Whether Mario is on ground
- `direction`: Mario's facing direction

### Enhanced State (20-feature mode)
All basic state features plus:
- `closest_enemy_distance`: Distance to nearest enemy
- `enemy_count`: Number of active enemies
- `powerup_present`: Whether a power-up is present
- `powerup_distance`: Distance to nearest power-up
- `solid_tiles_ahead`: Number of solid tiles ahead
- `pit_detected`: Whether a pit is detected ahead
- `velocity_magnitude`: Mario's velocity magnitude
- `facing_direction`: Mario's facing direction
- `mario_below_viewport`: Whether Mario fell below screen

## Reward Component Details

### Power-up Collection Rewards
- Triggered when `power_state` increases from previous frame
- Default reward: +50 points
- Tracks collection statistics

### Enemy Elimination Rewards
- Triggered when `enemy_count` decreases from previous frame
- Default reward: +25 points per enemy eliminated
- Tracks elimination statistics

### Environmental Navigation Rewards
- **Pit Avoidance**: +10 points when `pit_detected=True` and moving forward
- **Obstacle Navigation**: +5 points when `solid_tiles_ahead > 0` and moving forward
- Prioritizes pit avoidance over obstacle navigation

### Velocity Movement Rewards
- Rewards high forward velocity when `mario_x_vel > forward_momentum_threshold`
- Reward = `velocity_magnitude * velocity_movement_multiplier`
- Encourages maintaining speed and momentum

### Strategic Positioning Rewards
- Rewards maintaining safe distance from enemies
- Triggered when `closest_enemy_distance > safe_distance_threshold`
- Provides sustained bonus for extended safe positioning

### Enhanced Death Penalties
- Analyzes death cause using enhanced state information
- Applies appropriate penalty based on specific death type
- Helps AI learn to avoid specific dangerous situations

## Testing

Run the comprehensive test suite to verify the enhanced reward system:

```bash
python test_enhanced_rewards.py
```

The test suite covers:
- Basic reward calculation (12-feature mode)
- All enhanced reward components
- Death penalty system
- Backward compatibility
- Reward component logging and statistics

## Performance Considerations

- Enhanced reward calculation adds minimal computational overhead
- State tracking variables are efficiently managed
- Reward components are calculated only when enhanced features are enabled
- Statistics are updated incrementally for optimal performance

## Training Recommendations

1. **Gradual Introduction**: Start with basic rewards and gradually enable enhanced features
2. **Weight Tuning**: Adjust feature weights based on training progress and desired behaviors
3. **Monitoring**: Use enhanced statistics to monitor AI learning progress
4. **Balance**: Ensure enhanced rewards don't overshadow basic progression rewards

## Troubleshooting

### Common Issues

1. **Enhanced rewards not triggering**: Ensure `enhanced_features=True` and state contains required fields
2. **Unexpected reward values**: Check feature weights and configuration values
3. **Statistics not updating**: Verify enhanced state tracking is properly initialized

### Debug Logging

Enable debug logging to monitor reward calculations:

```python
import logging
logging.getLogger('python.environment.reward_calculator').setLevel(logging.DEBUG)
```

This will log detailed information about reward component calculations and state changes.