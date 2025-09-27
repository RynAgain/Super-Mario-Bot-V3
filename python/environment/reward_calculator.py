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
    
    @property
    def total(self) -> float:
        """Calculate total reward."""
        return (self.distance_reward + self.completion_reward + self.powerup_reward +
                self.enemy_reward + self.score_reward + self.coin_reward +
                self.death_penalty + self.movement_penalty + self.stuck_penalty)
    
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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize reward calculator.
        
        Args:
            config: Reward configuration parameters
        """
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
            }
        }
        
        # Update with provided config
        if config:
            self._update_config(self.config, config)
        
        # State tracking
        self.previous_state: Optional[Dict[str, Any]] = None
        self.max_x_reached = 0
        self.frames_stuck = 0
        self.last_x_position = 0
        self.progress_milestones_reached = set()
        
        # Episode tracking
        self.episode_start_time = None
        self.episode_rewards: List[float] = []
        
        # Statistics
        self.reward_stats = {
            'total_episodes': 0,
            'avg_episode_reward': 0.0,
            'max_episode_reward': float('-inf'),
            'min_episode_reward': float('inf'),
            'reward_components_avg': {}
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
        
        self.logger.debug("Episode reset")
    
    def calculate_frame_reward(self, current_state: Dict[str, Any]) -> Tuple[float, RewardComponents]:
        """
        Calculate reward for current frame based on state changes.
        
        Args:
            current_state: Current game state
            
        Returns:
            Tuple of (total_reward, reward_components)
        """
        if self.previous_state is None:
            self.reset_episode(current_state)
            return 0.0, RewardComponents()
        
        components = RewardComponents()
        
        # 1. Distance-based rewards (Primary - 60%)
        components.distance_reward = self._calculate_distance_reward(
            current_state.get('mario_x', 0),
            self.previous_state.get('mario_x', 0)
        )
        
        # 2. Power-up rewards (Secondary)
        components.powerup_reward = self._calculate_powerup_reward(
            self.previous_state.get('power_state', 0),
            current_state.get('power_state', 0)
        )
        
        # 3. Score and coin rewards (Secondary)
        score_increase = current_state.get('score', 0) - self.previous_state.get('score', 0)
        coins_collected = current_state.get('coins', 0) - self.previous_state.get('coins', 0)
        components.score_reward, components.coin_reward = self._calculate_score_rewards(
            score_increase, coins_collected
        )
        
        # 4. Enemy elimination rewards (Secondary)
        # Note: This would require enemy tracking, for now we estimate from score changes
        components.enemy_reward = self._estimate_enemy_reward(score_increase)
        
        # 5. Movement penalties
        components.movement_penalty = self._calculate_movement_penalty(
            current_state.get('mario_x', 0),
            self.previous_state.get('mario_x', 0)
        )
        
        # 6. Stuck penalty
        components.stuck_penalty = self._calculate_stuck_penalty(current_state.get('mario_x', 0))
        
        # Update tracking variables
        self.max_x_reached = max(self.max_x_reached, current_state.get('mario_x', 0))
        self._update_stuck_counter(current_state.get('mario_x', 0))
        self.previous_state = current_state.copy()
        
        # Apply reward shaping
        total_reward = self._apply_reward_shaping(components.total)
        
        # Track episode rewards
        self.episode_rewards.append(total_reward)
        
        return total_reward, components
    
    def calculate_episode_end_reward(self, episode_data: Dict[str, Any]) -> Tuple[float, RewardComponents]:
        """
        Calculate end-of-episode rewards and penalties.
        
        Args:
            episode_data: Episode completion data
            
        Returns:
            Tuple of (total_reward, reward_components)
        """
        components = RewardComponents()
        
        # Level completion reward
        if episode_data.get('level_completed', False):
            components.completion_reward = self._calculate_completion_reward(
                True, episode_data.get('time_remaining', 0)
            )
        
        # Death penalty
        if episode_data.get('died', False):
            components.death_penalty = self._calculate_death_penalty(
                episode_data.get('death_cause', DeathCause.UNKNOWN),
                episode_data.get('lives_remaining', 0)
            )
        
        # Distance achievement bonus
        final_distance = episode_data.get('max_x_reached', self.max_x_reached)
        components.distance_reward = final_distance * 0.5  # Bonus for total distance
        
        # Apply reward shaping
        total_reward = self._apply_reward_shaping(components.total)
        
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