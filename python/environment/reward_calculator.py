"""
Reward calculator for Super Mario Bros AI training system.

Implements the reward system based on level progression, survival behaviors,
and penalty system as specified in the reward system design.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class DeathCause(Enum):
    """Enumeration of death causes."""
    ENEMY_CONTACT = "enemy_contact"
    FALL_DEATH = "fall_death"
    TIMEOUT = "timeout"
    LAVA = "lava"
    UNKNOWN = "unknown"


class KillMethod(Enum):
    """Enumeration of enemy kill methods."""
    STOMP = "stomp"
    FIREBALL = "fireball"
    SHELL = "shell"
    STAR = "star"
    UNKNOWN = "unknown"


@dataclass
class RewardComponents:
    """Container for individual reward components."""
    distance_reward: float = 0.0
    completion_reward: float = 0.0
    powerup_reward: float = 0.0
    enemy_reward: float = 0.0
    score_reward: float = 0.0
    coin_reward: float = 0.0
    death_penalty: float = 0.0
    movement_penalty: float = 0.0
    stuck_penalty: float = 0.0
    
    # Enhanced reward components
    powerup_collection_reward: float = 0.0
    enemy_elimination_reward: float = 0.0
    environmental_navigation_reward: float = 0.0
    velocity_movement_reward: float = 0.0
    strategic_positioning_reward: float = 0.0
    enhanced_death_penalty: float = 0.0
    
    @property
    def total(self) -> float:
        """Calculate total reward."""
        return (self.distance_reward + self.completion_reward + self.powerup_reward +
                self.enemy_reward + self.score_reward + self.coin_reward +
                self.death_penalty + self.movement_penalty + self.stuck_penalty +
                self.powerup_collection_reward + self.enemy_elimination_reward +
                self.environmental_navigation_reward + self.velocity_movement_reward +
                self.strategic_positioning_reward + self.enhanced_death_penalty)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'distance_reward': self.distance_reward,
            'completion_reward': self.completion_reward,
            'powerup_reward': self.powerup_reward,
            'enemy_reward': self.enemy_reward,
            'score_reward': self.score_reward,
            'coin_reward': self.coin_reward,
            'death_penalty': self.death_penalty,
            'movement_penalty': self.movement_penalty,
            'stuck_penalty': self.stuck_penalty,
            'powerup_collection_reward': self.powerup_collection_reward,
            'enemy_elimination_reward': self.enemy_elimination_reward,
            'environmental_navigation_reward': self.environmental_navigation_reward,
            'velocity_movement_reward': self.velocity_movement_reward,
            'strategic_positioning_reward': self.strategic_positioning_reward,
            'enhanced_death_penalty': self.enhanced_death_penalty,
            'total': self.total
        }


class RewardCalculator:
    """
    Calculates rewards for Mario AI training based on game state changes.
    
    Implements the reward system with:
    - Primary rewards (70%): Distance progress and level completion
    - Secondary rewards (25%): Power-ups, enemy elimination, coins, score
    - Penalties (5%): Death, backward movement, getting stuck
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, enhanced_features: bool = False):
        """
        Initialize reward calculator.
        
        Args:
            config: Reward configuration parameters
            enhanced_features: Whether to use enhanced 20-feature reward calculation
        """
        self.enhanced_features = enhanced_features
        
        # Default configuration
        self.config = {
            'primary': {
                'forward_movement_multiplier': 1.0,
                'milestone_multiplier': 10.0,
                'completion_reward': 5000,
                'progress_bonuses': [100, 200, 300, 500]  # 25%, 50%, 75%, 90%
            },
            'secondary': {
                'powerup_rewards': [0, 200, 400],  # small, big, fire
                'enemy_kill_reward': 100,
                'coin_reward': 50,
                'score_multiplier': 0.01
            },
            'penalties': {
                'death_penalty': -1000,
                'backward_movement_multiplier': -0.5,
                'stuck_penalty_per_frame': -1.0,
                'stuck_threshold_frames': 60
            },
            'shaping': {
                'curriculum_enabled': True,
                'adaptive_scaling': True,
                'reward_clipping': [-2000, 2000]
            },
            # Enhanced reward configuration
            'enhanced': {
                'enabled': enhanced_features,
                'powerup_collection_reward': 50.0,
                'enemy_elimination_reward': 25.0,
                'environmental_navigation_reward': 5.0,
                'velocity_movement_multiplier': 0.1,
                'strategic_positioning_reward': 2.0,
                'safe_distance_threshold': 100.0,
                'pit_avoidance_reward': 10.0,
                'obstacle_navigation_reward': 5.0,
                'forward_momentum_threshold': 10.0,
                # Enhanced death penalties
                'pit_death_penalty': -100.0,
                'enemy_collision_penalty': -50.0,
                'time_death_penalty': -25.0,
                'general_death_penalty': -10.0,
                # Feature weights
                'feature_weights': {
                    'powerup_collection': 1.0,
                    'enemy_elimination': 1.0,
                    'environmental_awareness': 1.0,
                    'velocity_movement': 1.0,
                    'strategic_positioning': 1.0
                }
            }
        }
        
        # Update with provided config
        if config:
            self._update_config(self.config, config)
            
            # Load enhanced reward configuration if available
            if 'enhanced_rewards' in config and enhanced_features:
                enhanced_config = config['enhanced_rewards']
                self.config['enhanced'].update({
                    'enabled': enhanced_config.get('enabled', enhanced_features),
                    'powerup_collection_reward': enhanced_config.get('powerup_collection_reward', 50.0),
                    'enemy_elimination_reward': enhanced_config.get('enemy_elimination_reward', 25.0),
                    'environmental_navigation_reward': enhanced_config.get('environmental_navigation_reward', 5.0),
                    'velocity_movement_multiplier': enhanced_config.get('velocity_movement_multiplier', 0.1),
                    'strategic_positioning_reward': enhanced_config.get('strategic_positioning_reward', 2.0),
                    'safe_distance_threshold': enhanced_config.get('safe_distance_threshold', 100.0),
                    'pit_avoidance_reward': enhanced_config.get('pit_avoidance_reward', 10.0),
                    'obstacle_navigation_reward': enhanced_config.get('obstacle_navigation_reward', 5.0),
                    'forward_momentum_threshold': enhanced_config.get('forward_momentum_threshold', 10.0),
                    'pit_death_penalty': enhanced_config.get('pit_death_penalty', -100.0),
                    'enemy_collision_penalty': enhanced_config.get('enemy_collision_penalty', -50.0),
                    'time_death_penalty': enhanced_config.get('time_death_penalty', -25.0),
                    'general_death_penalty': enhanced_config.get('general_death_penalty', -10.0),
                    'feature_weights': enhanced_config.get('feature_weights', {
                        'powerup_collection': 1.0,
                        'enemy_elimination': 1.0,
                        'environmental_awareness': 1.0,
                        'velocity_movement': 1.0,
                        'strategic_positioning': 1.0
                    })
                })
        
        # State tracking
        self.previous_state: Optional[Dict[str, Any]] = None
        self.max_x_reached = 0
        self.frames_stuck = 0
        self.last_x_position = 0
        self.progress_milestones_reached = set()
        
        # Enhanced state tracking
        self.previous_power_state = 0
        self.previous_enemy_count = 0
        self.previous_powerup_present = False
        self.last_safe_distance_frames = 0
        
        # Episode tracking
        self.episode_start_time = None
        self.episode_rewards: List[float] = []
        
        # Statistics
        self.reward_stats = {
            'total_episodes': 0,
            'avg_episode_reward': 0.0,
            'max_episode_reward': float('-inf'),
            'min_episode_reward': float('inf'),
            'reward_components_avg': {},
            'enhanced_reward_stats': {
                'powerup_collections': 0,
                'enemy_eliminations': 0,
                'pit_avoidances': 0,
                'safe_positioning_frames': 0
            }
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _update_config(self, base_config: Dict, update_config: Dict):
        """Recursively update configuration."""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def reset_episode(self, initial_state: Dict[str, Any]):
        """
        Reset for new episode.
        
        Args:
            initial_state: Initial game state
        """
        self.previous_state = initial_state
        self.max_x_reached = initial_state.get('mario_x', 0)
        self.frames_stuck = 0
        self.last_x_position = initial_state.get('mario_x', 0)
        self.progress_milestones_reached.clear()
        self.episode_start_time = initial_state.get('timestamp', 0)
        self.episode_rewards.clear()
        
        # Reset enhanced state tracking
        if self.enhanced_features:
            self.previous_power_state = initial_state.get('power_state', 0)
            self.previous_enemy_count = initial_state.get('enemy_count', 0)
            self.previous_powerup_present = initial_state.get('powerup_present', False)
            self.last_safe_distance_frames = 0
        
        self.logger.debug("Episode reset")
    
    def calculate_frame_reward(self, current_state: Dict[str, Any]) -> Tuple[float, RewardComponents]:
        """
        Calculate reward for current frame with enhanced features support.
        
        Args:
            current_state: Current game state
            
        Returns:
            Tuple of (total_reward, reward_components)
        """
        if self.previous_state is None:
            self.reset_episode(current_state)
            return 0.0, RewardComponents()
        
        components = RewardComponents()
        current_x = current_state.get('mario_x', 0)
        previous_x = self.previous_state.get('mario_x', 0)
        
        # PRIMARY REWARD: New maximum distance (big reward)
        if current_x > self.max_x_reached:
            new_distance = current_x - self.max_x_reached
            components.distance_reward = new_distance * 1.0  # 1 point per pixel of new progress
            self.max_x_reached = current_x
        
        # SECONDARY REWARD: Any rightward movement (small reward to encourage exploration)
        elif current_x > previous_x:
            rightward_movement = current_x - previous_x
            components.distance_reward = rightward_movement * 0.1  # 0.1 points per pixel of rightward movement
        
        # SMALL PENALTY: Leftward movement (discourage going backward)
        elif current_x < previous_x:
            leftward_movement = previous_x - current_x
            components.movement_penalty = -leftward_movement * 0.05  # Small penalty for going backward
        
        # No reward/penalty for staying in same position (0.0)
        
        # Death detection and penalty
        if current_state.get('lives', 3) < self.previous_state.get('lives', 3):
            if self.enhanced_features:
                components.enhanced_death_penalty = self._calculate_enhanced_death_penalty(current_state)
            else:
                components.death_penalty = -50.0  # Small penalty, not too harsh
        
        # Enhanced reward calculations (only if enhanced features are enabled)
        if self.enhanced_features and self.config['enhanced']['enabled']:
            self._calculate_enhanced_rewards(current_state, components)
        
        # Update tracking variables
        self._update_stuck_counter(current_x)
        self.previous_state = current_state.copy()
        
        # Update enhanced state tracking
        if self.enhanced_features:
            self._update_enhanced_state_tracking(current_state)
        
        # Total reward includes all components
        total_reward = components.total
        
        # Track episode rewards
        self.episode_rewards.append(total_reward)
        
        return total_reward, components
    
    def calculate_episode_end_reward(self, episode_data: Dict[str, Any]) -> Tuple[float, RewardComponents]:
        """
        Calculate end-of-episode rewards based on maximum distance achieved.
        
        Args:
            episode_data: Episode completion data
            
        Returns:
            Tuple of (total_reward, reward_components)
        """
        components = RewardComponents()
        
        # SIMPLIFIED: Reward is just the maximum distance reached in this episode
        final_distance = episode_data.get('max_x_reached', self.max_x_reached)
        components.distance_reward = final_distance * 0.1  # 0.1 points per pixel reached
        
        # Big bonus for level completion
        if episode_data.get('level_completed', False):
            components.completion_reward = 1000.0  # Large completion bonus
        
        # Simple total reward
        total_reward = components.distance_reward + components.completion_reward
        
        # Update statistics
        self._update_episode_stats(total_reward)
        
        return total_reward, components
    
    def _calculate_distance_reward(self, current_x: int, previous_x: int) -> float:
        """Calculate distance-based reward."""
        # Immediate forward movement reward
        forward_movement = max(0, current_x - previous_x)
        movement_reward = forward_movement * self.config['primary']['forward_movement_multiplier']
        
        # Milestone rewards for new maximum distance
        milestone_reward = 0
        if current_x > self.max_x_reached:
            new_distance = current_x - self.max_x_reached
            # Exponential reward for reaching new areas
            milestone_reward = new_distance * self.config['primary']['milestone_multiplier']
            
            # Bonus for significant progress milestones
            progress_percentage = current_x / 3168.0  # World 1-1 length
            bonuses = self.config['primary']['progress_bonuses']
            
            milestones = [0.25, 0.50, 0.75, 0.90]
            for i, threshold in enumerate(milestones):
                if progress_percentage >= threshold and threshold not in self.progress_milestones_reached:
                    milestone_reward += bonuses[i] if i < len(bonuses) else 0
                    self.progress_milestones_reached.add(threshold)
        
        return movement_reward + milestone_reward
    
    def _calculate_completion_reward(self, level_completed: bool, time_remaining: int) -> float:
        """Calculate level completion reward."""
        if level_completed:
            base_completion = self.config['primary']['completion_reward']
            # Time bonus for faster completion
            time_bonus = time_remaining * 2.0
            return base_completion + time_bonus
        return 0
    
    def _calculate_powerup_reward(self, previous_power: int, current_power: int) -> float:
        """Calculate power-up collection reward."""
        power_values = self.config['secondary']['powerup_rewards']
        
        if current_power > previous_power and current_power < len(power_values):
            # Reward for gaining power
            return power_values[current_power] - power_values[previous_power]
        elif current_power < previous_power:
            # Penalty for losing power (but less than death)
            return -100
        return 0
    
    def _calculate_score_rewards(self, score_increase: int, coins_collected: int) -> Tuple[float, float]:
        """Calculate score and coin rewards."""
        # Small reward for score increase
        score_reward = score_increase * self.config['secondary']['score_multiplier']
        
        # Larger reward for coin collection
        coin_reward = coins_collected * self.config['secondary']['coin_reward']
        
        # Bonus for coin milestones
        if coins_collected >= 10:  # 1-up threshold
            coin_reward += 100
        
        return score_reward, coin_reward
    
    def _estimate_enemy_reward(self, score_increase: int) -> float:
        """Estimate enemy elimination reward from score changes."""
        # Common enemy point values in Super Mario Bros:
        # Goomba: 100, Koopa: 100, etc.
        if score_increase >= 100:
            estimated_kills = score_increase // 100
            return estimated_kills * self.config['secondary']['enemy_kill_reward']
        return 0
    
    def _calculate_death_penalty(self, death_cause: DeathCause, lives_remaining: int) -> float:
        """Calculate death penalty."""
        base_penalty = self.config['penalties']['death_penalty']
        
        # Adjust penalty based on cause
        cause_multiplier = {
            DeathCause.ENEMY_CONTACT: 1.0,
            DeathCause.FALL_DEATH: 1.2,     # Falling is worse
            DeathCause.TIMEOUT: 0.8,        # Less harsh for timeout
            DeathCause.LAVA: 1.5            # Lava death is worst
        }
        
        # Reduce penalty if many lives remaining (early in training)
        life_multiplier = max(0.5, 1.0 - (lives_remaining * 0.1))
        
        multiplier = cause_multiplier.get(death_cause, 1.0)
        return base_penalty * multiplier * life_multiplier
    
    def _calculate_movement_penalty(self, current_x: int, previous_x: int) -> float:
        """Calculate movement penalty for backward movement."""
        penalty = 0
        
        # Backward movement penalty
        if current_x < previous_x:
            backward_distance = previous_x - current_x
            penalty = backward_distance * self.config['penalties']['backward_movement_multiplier']
        
        return penalty
    
    def _calculate_stuck_penalty(self, current_x: int) -> float:
        """Calculate penalty for being stuck."""
        stuck_threshold = self.config['penalties']['stuck_threshold_frames']
        
        if self.frames_stuck > stuck_threshold:
            penalty_frames = min(self.frames_stuck - stuck_threshold, 120)  # Cap at 2 seconds
            return penalty_frames * self.config['penalties']['stuck_penalty_per_frame']
        
        return 0
    
    def _update_stuck_counter(self, current_x: int):
        """Update stuck frame counter."""
        if abs(current_x - self.last_x_position) < 2:  # Less than 2 pixels movement
            self.frames_stuck += 1
        else:
            self.frames_stuck = 0
        
        self.last_x_position = current_x
    
    def _apply_reward_shaping(self, reward: float) -> float:
        """Apply reward shaping techniques."""
        if not self.config['shaping']['adaptive_scaling']:
            return self._clip_reward(reward)
        
        # Apply curriculum learning if enabled
        if self.config['shaping']['curriculum_enabled']:
            reward = self._apply_curriculum_shaping(reward)
        
        return self._clip_reward(reward)
    
    def _apply_curriculum_shaping(self, reward: float) -> float:
        """Apply curriculum learning reward shaping."""
        # Simple curriculum: emphasize exploration early, completion later
        if self.reward_stats['total_episodes'] < 1000:
            # Early training: emphasize exploration
            if reward > 0:
                reward *= 1.2
            else:
                reward *= 0.8
        elif self.reward_stats['total_episodes'] > 5000:
            # Late training: emphasize consistency
            recent_avg = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else 0
            if recent_avg > 0:
                reward *= 0.9  # Reduce reward magnitude for consistency
        
        return reward
    
    def _clip_reward(self, reward: float) -> float:
        """Clip reward to configured bounds."""
        min_reward, max_reward = self.config['shaping']['reward_clipping']
        return np.clip(reward, min_reward, max_reward)
    
    def _update_episode_stats(self, episode_reward: float):
        """Update episode statistics."""
        self.reward_stats['total_episodes'] += 1
        
        # Update running averages
        n = self.reward_stats['total_episodes']
        self.reward_stats['avg_episode_reward'] = (
            (self.reward_stats['avg_episode_reward'] * (n - 1) + episode_reward) / n
        )
        
        # Update extremes
        self.reward_stats['max_episode_reward'] = max(
            self.reward_stats['max_episode_reward'], episode_reward
        )
        self.reward_stats['min_episode_reward'] = min(
            self.reward_stats['min_episode_reward'], episode_reward
        )
    
    def get_reward_stats(self) -> Dict[str, Any]:
        """Get reward calculation statistics."""
        stats = self.reward_stats.copy()
        stats.update({
            'current_max_x': self.max_x_reached,
            'frames_stuck': self.frames_stuck,
            'progress_milestones': list(self.progress_milestones_reached),
            'episode_reward_count': len(self.episode_rewards),
            'current_episode_total': sum(self.episode_rewards)
        })
        return stats
    
    def analyze_reward_effectiveness(self) -> Dict[str, Any]:
        """Analyze reward system effectiveness."""
        if len(self.episode_rewards) < 10:
            return {'insufficient_data': True}
        
        rewards = np.array(self.episode_rewards)
        
        return {
            'reward_variance': float(np.var(rewards)),
            'reward_trend': float(np.polyfit(range(len(rewards)), rewards, 1)[0]),
            'positive_reward_ratio': float(np.mean(rewards > 0)),
            'reward_stability': float(1.0 / (1.0 + np.std(rewards))),
            'learning_progress': float(np.mean(rewards[-10:]) - np.mean(rewards[:10]))
        }
    
    def detect_terminal_state(self, current_state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Detect if current state is terminal (episode should end).
        
        Args:
            current_state: Current game state
            
        Returns:
            Tuple of (is_terminal, reason)
        """
        # Death detection
        if current_state.get('lives', 3) < self.previous_state.get('lives', 3):
            return True, "death"
        
        # Level completion detection
        if current_state.get('level_progress', 0) >= 100:
            return True, "level_complete"
        
        # Timeout detection
        if current_state.get('time_remaining', 400) <= 0:
            return True, "timeout"
        
        # Stuck for too long (optional termination)
        if self.frames_stuck > 600:  # 10 seconds at 60 FPS
            return True, "stuck_timeout"
        
        return False, ""
    
    def get_config(self) -> Dict[str, Any]:
        """Get current reward configuration."""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update reward configuration."""
        self._update_config(self.config, new_config)
        self.logger.info("Reward configuration updated")
    
    def _calculate_enhanced_rewards(self, current_state: Dict[str, Any], components: RewardComponents):
        """
        Calculate enhanced reward components using 20-feature state information.
        
        Args:
            current_state: Current game state with enhanced features
            components: RewardComponents object to update
        """
        # Power-up collection reward
        components.powerup_collection_reward = self._calculate_powerup_collection_reward(current_state)
        
        # Enemy elimination reward
        components.enemy_elimination_reward = self._calculate_enemy_elimination_reward(current_state)
        
        # Environmental navigation reward
        components.environmental_navigation_reward = self._calculate_environmental_navigation_reward(current_state)
        
        # Velocity-based movement reward
        components.velocity_movement_reward = self._calculate_velocity_movement_reward(current_state)
        
        # Strategic positioning reward
        components.strategic_positioning_reward = self._calculate_strategic_positioning_reward(current_state)
        
        # Apply feature weights
        weights = self.config['enhanced']['feature_weights']
        components.powerup_collection_reward *= weights.get('powerup_collection', 1.0)
        components.enemy_elimination_reward *= weights.get('enemy_elimination', 1.0)
        components.environmental_navigation_reward *= weights.get('environmental_awareness', 1.0)
        components.velocity_movement_reward *= weights.get('velocity_movement', 1.0)
        components.strategic_positioning_reward *= weights.get('strategic_positioning', 1.0)
    
    def _calculate_powerup_collection_reward(self, current_state: Dict[str, Any]) -> float:
        """
        Calculate reward for power-up collection.
        
        Args:
            current_state: Current game state
            
        Returns:
            Power-up collection reward
        """
        reward = 0.0
        
        # Check for power state increase (power-up collected)
        current_power = current_state.get('power_state', 0)
        if current_power > self.previous_power_state:
            reward = self.config['enhanced']['powerup_collection_reward']
            self.reward_stats['enhanced_reward_stats']['powerup_collections'] += 1
            self.logger.debug(f"Power-up collected! Power state: {self.previous_power_state} -> {current_power}")
        
        return reward
    
    def _calculate_enemy_elimination_reward(self, current_state: Dict[str, Any]) -> float:
        """
        Calculate reward for enemy elimination.
        
        Args:
            current_state: Current game state
            
        Returns:
            Enemy elimination reward
        """
        reward = 0.0
        
        # Check for enemy count decrease (enemy eliminated)
        current_enemy_count = current_state.get('enemy_count', 0)
        if current_enemy_count < self.previous_enemy_count:
            enemies_eliminated = self.previous_enemy_count - current_enemy_count
            reward = enemies_eliminated * self.config['enhanced']['enemy_elimination_reward']
            self.reward_stats['enhanced_reward_stats']['enemy_eliminations'] += enemies_eliminated
            self.logger.debug(f"Enemy eliminated! Count: {self.previous_enemy_count} -> {current_enemy_count}")
        
        return reward
    
    def _calculate_environmental_navigation_reward(self, current_state: Dict[str, Any]) -> float:
        """
        Calculate reward for environmental navigation (pit avoidance, obstacle navigation).
        
        Args:
            current_state: Current game state
            
        Returns:
            Environmental navigation reward
        """
        reward = 0.0
        mario_x_vel = current_state.get('mario_x_vel', 0)
        
        # Pit avoidance reward (higher priority)
        if current_state.get('pit_detected', False):
            # If pit is detected and Mario is moving forward, reward avoidance
            if mario_x_vel > 0:  # Moving forward despite pit
                reward += self.config['enhanced']['pit_avoidance_reward']
                self.reward_stats['enhanced_reward_stats']['pit_avoidances'] += 1
                self.logger.debug("Pit avoidance reward applied")
                return reward  # Return early to avoid double rewards
        
        # Obstacle navigation reward (only if no pit detected)
        solid_tiles_ahead = current_state.get('solid_tiles_ahead', 0)
        if solid_tiles_ahead > 0:
            # Reward for navigating through areas with obstacles
            if mario_x_vel > 0:  # Moving forward through obstacles
                reward += self.config['enhanced']['obstacle_navigation_reward']
                self.logger.debug("Obstacle navigation reward applied")
        
        return reward
    
    def _calculate_velocity_movement_reward(self, current_state: Dict[str, Any]) -> float:
        """
        Calculate reward for velocity-based movement (forward momentum).
        
        Args:
            current_state: Current game state
            
        Returns:
            Velocity movement reward
        """
        reward = 0.0
        
        # Forward momentum reward
        velocity_magnitude = current_state.get('velocity_magnitude', 0.0)
        mario_x_vel = current_state.get('mario_x_vel', 0)
        
        if mario_x_vel > self.config['enhanced']['forward_momentum_threshold']:
            # Reward high forward velocity
            reward = velocity_magnitude * self.config['enhanced']['velocity_movement_multiplier']
            self.logger.debug(f"Forward momentum reward: {reward:.2f} (velocity: {velocity_magnitude:.2f})")
        
        return reward
    
    def _calculate_strategic_positioning_reward(self, current_state: Dict[str, Any]) -> float:
        """
        Calculate reward for strategic positioning (maintaining safe distance from enemies).
        
        Args:
            current_state: Current game state
            
        Returns:
            Strategic positioning reward
        """
        reward = 0.0
        
        # Safe distance from enemies reward
        closest_enemy_distance = current_state.get('closest_enemy_distance', 999.0)
        safe_distance_threshold = self.config['enhanced']['safe_distance_threshold']
        
        if closest_enemy_distance > safe_distance_threshold:
            # Reward for maintaining safe distance
            reward = self.config['enhanced']['strategic_positioning_reward']
            self.last_safe_distance_frames += 1
            self.reward_stats['enhanced_reward_stats']['safe_positioning_frames'] += 1
        else:
            self.last_safe_distance_frames = 0
        
        # Bonus for sustained safe positioning
        if self.last_safe_distance_frames > 60:  # 1 second at 60 FPS
            reward *= 1.5  # 50% bonus for sustained safe positioning
        
        return reward
    
    def _calculate_enhanced_death_penalty(self, current_state: Dict[str, Any]) -> float:
        """
        Calculate enhanced death penalty based on specific death causes.
        
        Args:
            current_state: Current game state
            
        Returns:
            Enhanced death penalty (negative value)
        """
        penalty = 0.0
        
        # Determine death cause based on enhanced state information
        if current_state.get('mario_below_viewport', False):
            # Pit death
            penalty = self.config['enhanced']['pit_death_penalty']
            self.logger.debug("Pit death penalty applied")
        elif current_state.get('closest_enemy_distance', 999.0) < 20.0:
            # Enemy collision death
            penalty = self.config['enhanced']['enemy_collision_penalty']
            self.logger.debug("Enemy collision death penalty applied")
        elif current_state.get('time_remaining', 400) <= 0:
            # Time-based death
            penalty = self.config['enhanced']['time_death_penalty']
            self.logger.debug("Time death penalty applied")
        else:
            # General death penalty
            penalty = self.config['enhanced']['general_death_penalty']
            self.logger.debug("General death penalty applied")
        
        return penalty
    
    def _update_enhanced_state_tracking(self, current_state: Dict[str, Any]):
        """
        Update enhanced state tracking variables.
        
        Args:
            current_state: Current game state
        """
        self.previous_power_state = current_state.get('power_state', 0)
        self.previous_enemy_count = current_state.get('enemy_count', 0)
        self.previous_powerup_present = current_state.get('powerup_present', False)
    
    def get_enhanced_reward_stats(self) -> Dict[str, Any]:
        """
        Get enhanced reward statistics.
        
        Returns:
            Dictionary containing enhanced reward statistics
        """
        if not self.enhanced_features:
            return {}
        
        stats = self.reward_stats['enhanced_reward_stats'].copy()
        stats.update({
            'safe_positioning_ratio': (
                stats['safe_positioning_frames'] / max(1, len(self.episode_rewards))
            ),
            'powerup_collection_rate': stats['powerup_collections'] / max(1, self.reward_stats['total_episodes']),
            'enemy_elimination_rate': stats['enemy_eliminations'] / max(1, self.reward_stats['total_episodes'])
        })
        
        return stats


# Utility functions for reward analysis

def analyze_reward_distribution(rewards: List[float]) -> Dict[str, float]:
    """Analyze reward distribution statistics."""
    if not rewards:
        return {}
    
    rewards_array = np.array(rewards)
    
    return {
        'mean': float(np.mean(rewards_array)),
        'std': float(np.std(rewards_array)),
        'min': float(np.min(rewards_array)),
        'max': float(np.max(rewards_array)),
        'median': float(np.median(rewards_array)),
        'q25': float(np.percentile(rewards_array, 25)),
        'q75': float(np.percentile(rewards_array, 75)),
        'positive_ratio': float(np.mean(rewards_array > 0)),
        'zero_ratio': float(np.mean(rewards_array == 0)),
        'negative_ratio': float(np.mean(rewards_array < 0))
    }


def plot_reward_components(reward_components_history: List[RewardComponents], 
                          save_path: Optional[str] = None):
    """
    Plot reward components over time (requires matplotlib).
    
    Args:
        reward_components_history: List of reward components
        save_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
        enable_plotting = False  # Keep plots off in training runs
        if not enable_plotting:
            return
        
        components = ['distance_reward', 'powerup_reward', 'score_reward', 
                     'death_penalty', 'movement_penalty']
        
        fig, axes = plt.subplots(len(components), 1, figsize=(12, 8), sharex=True)
        
        for i, component in enumerate(components):
            values = [getattr(rc, component) for rc in reward_components_history]
            axes[i].plot(values)
            axes[i].set_ylabel(component.replace('_', ' ').title())
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Frame')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib not available for plotting")