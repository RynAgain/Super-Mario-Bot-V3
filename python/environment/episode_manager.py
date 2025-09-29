"""
Episode manager for Super Mario Bros AI training system.

Handles episode boundaries, statistics tracking, curriculum learning progression,
and integration with CSV logging system.
"""

import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

from .reward_calculator import RewardCalculator, RewardComponents, DeathCause


class EpisodeStatus(Enum):
    """Episode status enumeration."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


class CurriculumPhase(Enum):
    """Curriculum learning phases."""
    EXPLORATION = "exploration"
    OPTIMIZATION = "optimization"
    MASTERY = "mastery"


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    episode_id: int
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    
    # Game progress
    max_x_reached: int = 0
    level_progress_percent: float = 0.0
    level_completed: bool = False
    
    # Survival stats
    lives_used: int = 0
    deaths: int = 0
    death_causes: List[str] = None
    
    # Performance metrics
    total_reward: float = 0.0
    avg_reward_per_frame: float = 0.0
    reward_components: Dict[str, float] = None
    
    # Game metrics
    score: int = 0
    coins_collected: int = 0
    enemies_killed: int = 0
    powerups_collected: int = 0
    time_remaining: int = 0
    
    # Technical metrics
    frames_processed: int = 0
    actions_taken: int = 0
    avg_fps: float = 0.0
    sync_quality: float = 0.0
    
    # Status
    status: EpisodeStatus = EpisodeStatus.NOT_STARTED
    termination_reason: str = ""
    
    def __post_init__(self):
        """Initialize mutable default values."""
        if self.death_causes is None:
            self.death_causes = []
        if self.reward_components is None:
            self.reward_components = {}
    
    def calculate_derived_stats(self):
        """Calculate derived statistics."""
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time
        
        if self.frames_processed > 0:
            self.avg_reward_per_frame = self.total_reward / self.frames_processed
            if self.duration and self.duration > 0:
                self.avg_fps = self.frames_processed / self.duration
        
        # Calculate level progress
        if self.max_x_reached > 0:
            self.level_progress_percent = min(100.0, (self.max_x_reached / 3168.0) * 100.0)


class EpisodeManager:
    """
    Manages training episodes, statistics, and curriculum learning progression.
    """
    
    def __init__(self, 
                 reward_calculator: RewardCalculator,
                 log_directory: str = "logs",
                 csv_filename: str = "training_episodes.csv"):
        """
        Initialize episode manager.
        
        Args:
            reward_calculator: Reward calculator instance
            log_directory: Directory for log files
            csv_filename: CSV log filename
        """
        self.reward_calculator = reward_calculator
        self.log_directory = Path(log_directory)
        self.csv_filename = csv_filename
        
        # Create log directory
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Current episode state
        self.current_episode: Optional[EpisodeStats] = None
        self.episode_counter = 0
        
        # Episode history
        self.episode_history: List[EpisodeStats] = []
        self.max_history_size = 1000
        
        # Curriculum learning
        self.curriculum_phase = CurriculumPhase.EXPLORATION
        self.curriculum_config = {
            'exploration_episodes': 1000,
            'optimization_episodes': 4000,
            'mastery_threshold': 0.8  # Success rate threshold for mastery
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'avg_episode_reward': 0.0,
            'avg_max_distance': 0.0,
            'level_completion_rate': 0.0,
            'recent_performance': deque(maxlen=100)  # Last 100 episodes
        }
        
        # CSV logging
        self.csv_file_path = self.log_directory / self.csv_filename
        self.csv_headers = self._get_csv_headers()
        self._initialize_csv_file()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Episode manager initialized with log directory: {self.log_directory}")
    
    def _get_csv_headers(self) -> List[str]:
        """Get CSV headers for episode logging."""
        return [
            'episode_id', 'timestamp', 'start_time', 'end_time', 'duration',
            'max_x_reached', 'level_progress_percent', 'level_completed',
            'lives_used', 'deaths', 'death_causes',
            'total_reward', 'avg_reward_per_frame',
            'distance_reward', 'completion_reward', 'powerup_reward',
            'enemy_reward', 'score_reward', 'coin_reward',
            'death_penalty', 'movement_penalty', 'stuck_penalty',
            'score', 'coins_collected', 'enemies_killed', 'powerups_collected',
            'time_remaining', 'frames_processed', 'actions_taken',
            'avg_fps', 'sync_quality', 'status', 'termination_reason',
            'curriculum_phase'
        ]
    
    def _initialize_csv_file(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.csv_file_path.exists():
            with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)
            self.logger.info(f"Created CSV log file: {self.csv_file_path}")
    
    def start_episode(self, initial_state: Dict[str, Any]) -> int:
        """
        Start a new episode.
        
        Args:
            initial_state: Initial game state
            
        Returns:
            Episode ID
        """
        # End current episode if running
        if self.current_episode and self.current_episode.status == EpisodeStatus.RUNNING:
            self.logger.warning("Starting new episode while previous episode is still running")
            self.end_episode({"termination_reason": "forced_restart"})
        
        # Create new episode
        self.episode_counter += 1
        self.current_episode = EpisodeStats(
            episode_id=self.episode_counter,
            start_time=time.time(),
            status=EpisodeStatus.RUNNING
        )
        
        # Initialize reward calculator
        self.reward_calculator.reset_episode(initial_state)
        
        # Update curriculum phase
        self._update_curriculum_phase()
        
        self.logger.info(f"Started episode {self.episode_counter} in {self.curriculum_phase.value} phase")
        
        return self.episode_counter
    
    def process_frame(self, 
                     game_state: Dict[str, Any], 
                     action_taken: Optional[Dict[str, bool]] = None,
                     sync_quality: float = 0.0) -> Tuple[float, RewardComponents, bool]:
        """
        Process a single frame during episode.
        
        Args:
            game_state: Current game state
            action_taken: Action taken this frame
            sync_quality: Frame synchronization quality
            
        Returns:
            Tuple of (frame_reward, reward_components, is_terminal)
        """
        if not self.current_episode or self.current_episode.status != EpisodeStatus.RUNNING:
            self.logger.warning("Processing frame without active episode")
            return 0.0, RewardComponents(), False
        
        # Calculate frame reward
        frame_reward, reward_components = self.reward_calculator.calculate_frame_reward(game_state)
        
        # Update episode stats
        self.current_episode.frames_processed += 1
        self.current_episode.total_reward += frame_reward
        self.current_episode.max_x_reached = max(
            self.current_episode.max_x_reached, 
            game_state.get('mario_x', 0)
        )
        
        if action_taken:
            self.current_episode.actions_taken += 1
        
        # Update sync quality (running average)
        if self.current_episode.frames_processed == 1:
            self.current_episode.sync_quality = sync_quality
        else:
            alpha = 0.1  # Smoothing factor
            self.current_episode.sync_quality = (
                (1 - alpha) * self.current_episode.sync_quality + alpha * sync_quality
            )
        
        # Check for terminal state
        is_terminal, termination_reason = self.reward_calculator.detect_terminal_state(game_state)
        
        if is_terminal:
            self.current_episode.termination_reason = termination_reason
            
            # Handle different termination reasons
            if termination_reason == "death":
                self.current_episode.deaths += 1
                self.current_episode.death_causes.append(termination_reason)
            elif termination_reason == "level_complete":
                self.current_episode.level_completed = True
                self.current_episode.status = EpisodeStatus.COMPLETED
            elif termination_reason == "timeout":
                self.current_episode.status = EpisodeStatus.FAILED
            else:
                self.current_episode.status = EpisodeStatus.TERMINATED
        
        return frame_reward, reward_components, is_terminal
    
    def end_episode(self, episode_data: Dict[str, Any]) -> EpisodeStats:
        """
        End current episode and calculate final statistics.
        
        Args:
            episode_data: Final episode data
            
        Returns:
            Completed episode statistics
        """
        if not self.current_episode:
            self.logger.warning("Attempting to end episode when none is active")
            return None
        
        # Set end time
        self.current_episode.end_time = time.time()
        
        # Calculate episode-end rewards
        episode_reward, episode_components = self.reward_calculator.calculate_episode_end_reward(episode_data)
        self.current_episode.total_reward += episode_reward
        
        # Update episode stats from final data
        self._update_episode_stats_from_data(episode_data)
        
        # Calculate derived statistics
        self.current_episode.calculate_derived_stats()
        
        # Update reward components
        self.current_episode.reward_components = episode_components.to_dict()
        
        # Set final status if not already set
        if self.current_episode.status == EpisodeStatus.RUNNING:
            if self.current_episode.level_completed:
                self.current_episode.status = EpisodeStatus.COMPLETED
            elif self.current_episode.deaths > 0:
                self.current_episode.status = EpisodeStatus.FAILED
            else:
                self.current_episode.status = EpisodeStatus.TERMINATED
        
        # Add to history
        self.episode_history.append(self.current_episode)
        if len(self.episode_history) > self.max_history_size:
            self.episode_history.pop(0)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Log to CSV
        self._log_episode_to_csv(self.current_episode)
        
        # Log episode summary
        self._log_episode_summary(self.current_episode)
        
        completed_episode = self.current_episode
        self.current_episode = None
        
        return completed_episode
    
    def _update_episode_stats_from_data(self, episode_data: Dict[str, Any]):
        """Update episode stats from final episode data."""
        self.current_episode.score = episode_data.get('final_score', 0)
        self.current_episode.time_remaining = episode_data.get('time_remaining', 0)
        self.current_episode.coins_collected = episode_data.get('coins_collected', 0)
        self.current_episode.lives_used = episode_data.get('lives_used', 0)
        
        # Estimate enemies killed from score (rough approximation)
        if self.current_episode.score > 0:
            self.current_episode.enemies_killed = max(0, (self.current_episode.score - 200) // 100)
        
        # Update termination reason if provided
        if 'termination_reason' in episode_data:
            self.current_episode.termination_reason = episode_data['termination_reason']
    
    def _update_performance_metrics(self):
        """Update overall performance metrics."""
        self.performance_metrics['total_episodes'] += 1
        
        if self.current_episode.status == EpisodeStatus.COMPLETED:
            self.performance_metrics['successful_episodes'] += 1
        
        # Update running averages
        n = self.performance_metrics['total_episodes']
        self.performance_metrics['avg_episode_reward'] = (
            (self.performance_metrics['avg_episode_reward'] * (n - 1) + 
             self.current_episode.total_reward) / n
        )
        
        self.performance_metrics['avg_max_distance'] = (
            (self.performance_metrics['avg_max_distance'] * (n - 1) + 
             self.current_episode.max_x_reached) / n
        )
        
        # Update completion rate
        self.performance_metrics['level_completion_rate'] = (
            self.performance_metrics['successful_episodes'] / n
        )
        
        # Add to recent performance tracking
        episode_success = 1.0 if self.current_episode.status == EpisodeStatus.COMPLETED else 0.0
        self.performance_metrics['recent_performance'].append(episode_success)
    
    def _update_curriculum_phase(self):
        """Update curriculum learning phase based on progress."""
        total_episodes = self.performance_metrics['total_episodes']
        
        if total_episodes < self.curriculum_config['exploration_episodes']:
            self.curriculum_phase = CurriculumPhase.EXPLORATION
        elif total_episodes < (self.curriculum_config['exploration_episodes'] + 
                              self.curriculum_config['optimization_episodes']):
            self.curriculum_phase = CurriculumPhase.OPTIMIZATION
        else:
            # Check if ready for mastery phase
            recent_success_rate = np.mean(list(self.performance_metrics['recent_performance']))
            if recent_success_rate >= self.curriculum_config['mastery_threshold']:
                self.curriculum_phase = CurriculumPhase.MASTERY
            else:
                self.curriculum_phase = CurriculumPhase.OPTIMIZATION
    
    def _log_episode_to_csv(self, episode: EpisodeStats):
        """Log episode to CSV file."""
        try:
            with open(self.csv_file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Prepare row data
                row = [
                    episode.episode_id,
                    datetime.fromtimestamp(episode.start_time).isoformat(),
                    episode.start_time,
                    episode.end_time,
                    episode.duration,
                    episode.max_x_reached,
                    episode.level_progress_percent,
                    episode.level_completed,
                    episode.lives_used,
                    episode.deaths,
                    ';'.join(episode.death_causes),
                    episode.total_reward,
                    episode.avg_reward_per_frame,
                    episode.reward_components.get('distance_reward', 0),
                    episode.reward_components.get('completion_reward', 0),
                    episode.reward_components.get('powerup_reward', 0),
                    episode.reward_components.get('enemy_reward', 0),
                    episode.reward_components.get('score_reward', 0),
                    episode.reward_components.get('coin_reward', 0),
                    episode.reward_components.get('death_penalty', 0),
                    episode.reward_components.get('movement_penalty', 0),
                    episode.reward_components.get('stuck_penalty', 0),
                    episode.score,
                    episode.coins_collected,
                    episode.enemies_killed,
                    episode.powerups_collected,
                    episode.time_remaining,
                    episode.frames_processed,
                    episode.actions_taken,
                    episode.avg_fps,
                    episode.sync_quality,
                    episode.status.value,
                    episode.termination_reason,
                    self.curriculum_phase.value
                ]
                
                writer.writerow(row)
                
        except Exception as e:
            self.logger.error(f"Failed to log episode to CSV: {e}")
    
    def _log_episode_summary(self, episode: EpisodeStats):
        """Log episode summary to logger."""
        self.logger.info(
            f"Episode {episode.episode_id} completed: "
            f"Status={episode.status.value}, "
            f"Reward={episode.total_reward:.1f}, "
            f"Distance={episode.max_x_reached}, "
            f"Progress={episode.level_progress_percent:.1f}%, "
            f"Duration={episode.duration:.1f}s, "
            f"FPS={episode.avg_fps:.1f}"
        )
    
    # Public interface methods
    
    def get_current_episode_stats(self) -> Optional[EpisodeStats]:
        """Get current episode statistics."""
        return self.current_episode
    
    def get_episode_history(self, limit: Optional[int] = None) -> List[EpisodeStats]:
        """Get episode history."""
        if limit:
            return self.episode_history[-limit:]
        return self.episode_history.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall performance metrics."""
        metrics = self.performance_metrics.copy()
        metrics['curriculum_phase'] = self.curriculum_phase.value
        metrics['recent_success_rate'] = (
            np.mean(list(self.performance_metrics['recent_performance'])) 
            if self.performance_metrics['recent_performance'] else 0.0
        )
        return metrics
    
    def get_curriculum_progress(self) -> Dict[str, Any]:
        """Get curriculum learning progress."""
        total_episodes = self.performance_metrics['total_episodes']
        
        return {
            'current_phase': self.curriculum_phase.value,
            'total_episodes': total_episodes,
            'exploration_progress': min(1.0, total_episodes / self.curriculum_config['exploration_episodes']),
            'optimization_progress': max(0.0, min(1.0, 
                (total_episodes - self.curriculum_config['exploration_episodes']) / 
                self.curriculum_config['optimization_episodes'])),
            'mastery_ready': (
                np.mean(list(self.performance_metrics['recent_performance'])) >= 
                self.curriculum_config['mastery_threshold']
                if self.performance_metrics['recent_performance'] else False
            )
        }
    
    def analyze_recent_performance(self, window_size: int = 100) -> Dict[str, Any]:
        """Analyze recent performance trends."""
        if len(self.episode_history) < window_size:
            recent_episodes = self.episode_history
        else:
            recent_episodes = self.episode_history[-window_size:]
        
        if not recent_episodes:
            return {'no_data': True}
        
        # Calculate metrics
        rewards = [ep.total_reward for ep in recent_episodes]
        distances = [ep.max_x_reached for ep in recent_episodes]
        completions = [1 if ep.level_completed else 0 for ep in recent_episodes]
        
        return {
            'window_size': len(recent_episodes),
            'avg_reward': np.mean(rewards),
            'reward_trend': np.polyfit(range(len(rewards)), rewards, 1)[0] if len(rewards) > 1 else 0,
            'avg_distance': np.mean(distances),
            'distance_trend': np.polyfit(range(len(distances)), distances, 1)[0] if len(distances) > 1 else 0,
            'completion_rate': np.mean(completions),
            'improvement_rate': (
                np.mean(completions[-window_size//4:]) - np.mean(completions[:window_size//4])
                if len(completions) >= window_size//2 else 0
            )
        }
    
    def export_episode_data(self, output_path: str, format: str = 'csv'):
        """
        Export episode data to file.
        
        Args:
            output_path: Output file path
            format: Export format ('csv' or 'json')
        """
        if format == 'csv':
            # Copy current CSV file
            import shutil
            shutil.copy2(self.csv_file_path, output_path)
        elif format == 'json':
            import json
            data = [asdict(episode) for episode in self.episode_history]
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported episode data to {output_path}")
    
    def reset_statistics(self):
        """Reset all statistics and history."""
        self.episode_history.clear()
        self.performance_metrics = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'avg_episode_reward': 0.0,
            'avg_max_distance': 0.0,
            'level_completion_rate': 0.0,
            'recent_performance': deque(maxlen=100)
        }
        self.curriculum_phase = CurriculumPhase.EXPLORATION
        self.logger.info("Episode statistics reset")


# Import deque for recent performance tracking
from collections import deque