"""
Training utilities for Super Mario Bros AI training system.

Provides training state management, checkpoint handling, performance evaluation,
and system health monitoring capabilities.
"""

import json
import logging
import pickle
import time
import torch
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np
import GPUtil


@dataclass
class TrainingState:
    """Training state data structure."""
    session_id: str
    episode: int
    step: int
    total_steps: int
    start_time: float
    current_time: float
    
    # Training progress
    total_episodes_completed: int
    successful_episodes: int
    best_episode_reward: float
    best_episode_distance: int
    
    # Current episode state
    current_episode_reward: float
    current_episode_steps: int
    current_episode_max_x: int
    
    # Agent state
    epsilon: float
    learning_rate: float
    replay_buffer_size: int
    
    # Performance metrics
    avg_episode_reward: float
    avg_episode_distance: float
    completion_rate: float
    recent_performance: List[float]
    
    # System state
    curriculum_phase: str
    training_phase: str  # warmup, training, evaluation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingState':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TrainingMetrics:
    """Training performance metrics."""
    episode_rewards: deque
    episode_distances: deque
    episode_durations: deque
    episode_completions: deque
    
    loss_history: deque
    q_value_history: deque
    epsilon_history: deque
    
    system_metrics: deque
    
    def __post_init__(self):
        """Initialize deques with maxlen if they're not already deques."""
        if not isinstance(self.episode_rewards, deque):
            self.episode_rewards = deque(self.episode_rewards, maxlen=1000)
        if not isinstance(self.episode_distances, deque):
            self.episode_distances = deque(self.episode_distances, maxlen=1000)
        if not isinstance(self.episode_durations, deque):
            self.episode_durations = deque(self.episode_durations, maxlen=1000)
        if not isinstance(self.episode_completions, deque):
            self.episode_completions = deque(self.episode_completions, maxlen=1000)
        if not isinstance(self.loss_history, deque):
            self.loss_history = deque(self.loss_history, maxlen=10000)
        if not isinstance(self.q_value_history, deque):
            self.q_value_history = deque(self.q_value_history, maxlen=10000)
        if not isinstance(self.epsilon_history, deque):
            self.epsilon_history = deque(self.epsilon_history, maxlen=10000)
        if not isinstance(self.system_metrics, deque):
            self.system_metrics = deque(self.system_metrics, maxlen=1000)
    
    def add_episode_metrics(self, reward: float, distance: int, duration: float, completed: bool):
        """Add episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_distances.append(distance)
        self.episode_durations.append(duration)
        self.episode_completions.append(1.0 if completed else 0.0)
    
    def add_training_metrics(self, loss: float, q_value: float, epsilon: float):
        """Add training step metrics."""
        self.loss_history.append(loss)
        self.q_value_history.append(q_value)
        self.epsilon_history.append(epsilon)
    
    def add_system_metrics(self, metrics: Dict[str, float]):
        """Add system performance metrics."""
        self.system_metrics.append(metrics)
    
    def get_recent_performance(self, window: int = 100) -> Dict[str, float]:
        """Get recent performance statistics."""
        if len(self.episode_rewards) < window:
            window = len(self.episode_rewards)
        
        if window == 0:
            return {}
        
        recent_rewards = list(self.episode_rewards)[-window:]
        recent_distances = list(self.episode_distances)[-window:]
        recent_completions = list(self.episode_completions)[-window:]
        
        return {
            'avg_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'max_reward': np.max(recent_rewards),
            'min_reward': np.min(recent_rewards),
            'avg_distance': np.mean(recent_distances),
            'max_distance': np.max(recent_distances),
            'completion_rate': np.mean(recent_completions),
            'improvement_trend': self._calculate_trend(recent_rewards)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate improvement trend using linear regression."""
        if len(values) < 10:
            return 0.0
        
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]  # Slope indicates trend


class TrainingStateManager:
    """
    Manages training state persistence, checkpointing, and recovery.
    """
    
    def __init__(self, 
                 checkpoint_dir: str = "checkpoints",
                 state_file: str = "training_state.json",
                 auto_save_interval: int = 100):
        """
        Initialize training state manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint files
            state_file: Training state file name
            auto_save_interval: Auto-save interval in steps
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.checkpoint_dir / state_file
        self.auto_save_interval = auto_save_interval
        
        # Current state
        self.training_state: Optional[TrainingState] = None
        self.training_metrics = TrainingMetrics(
            episode_rewards=deque(maxlen=1000),
            episode_distances=deque(maxlen=1000),
            episode_durations=deque(maxlen=1000),
            episode_completions=deque(maxlen=1000),
            loss_history=deque(maxlen=10000),
            q_value_history=deque(maxlen=10000),
            epsilon_history=deque(maxlen=10000),
            system_metrics=deque(maxlen=1000)
        )
        
        # Auto-save tracking
        self.last_auto_save = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Training state manager initialized: {self.checkpoint_dir}")
    
    def initialize_training_state(self, session_id: str, config: Dict[str, Any]) -> TrainingState:
        """
        Initialize new training state.
        
        Args:
            session_id: Training session identifier
            config: Training configuration
            
        Returns:
            Initialized training state
        """
        self.training_state = TrainingState(
            session_id=session_id,
            episode=0,
            step=0,
            total_steps=0,
            start_time=time.time(),
            current_time=time.time(),
            total_episodes_completed=0,
            successful_episodes=0,
            best_episode_reward=float('-inf'),
            best_episode_distance=0,
            current_episode_reward=0.0,
            current_episode_steps=0,
            current_episode_max_x=0,
            epsilon=config.get('epsilon_start', 1.0),
            learning_rate=config.get('learning_rate', 0.00025),
            replay_buffer_size=0,
            avg_episode_reward=0.0,
            avg_episode_distance=0.0,
            completion_rate=0.0,
            recent_performance=[],
            curriculum_phase="exploration",
            training_phase="warmup"
        )
        
        self.save_training_state()
        self.logger.info(f"Initialized training state for session: {session_id}")
        
        return self.training_state
    
    def load_training_state(self, session_id: Optional[str] = None) -> Optional[TrainingState]:
        """
        Load training state from file.
        
        Args:
            session_id: Optional session ID to load specific state
            
        Returns:
            Loaded training state or None if not found
        """
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                # Filter by session ID if provided
                if session_id and data.get('session_id') != session_id:
                    self.logger.warning(f"Session ID mismatch: expected {session_id}, found {data.get('session_id')}")
                    return None
                
                self.training_state = TrainingState.from_dict(data)
                self.logger.info(f"Loaded training state for session: {self.training_state.session_id}")
                
                return self.training_state
            
        except Exception as e:
            self.logger.error(f"Failed to load training state: {e}")
        
        return None
    
    def save_training_state(self):
        """Save current training state to file."""
        if not self.training_state:
            return
        
        try:
            self.training_state.current_time = time.time()
            
            with open(self.state_file, 'w') as f:
                json.dump(self.training_state.to_dict(), f, indent=2)
            
            self.logger.debug(f"Saved training state for episode {self.training_state.episode}")
            
        except Exception as e:
            self.logger.error(f"Failed to save training state: {e}")
    
    def update_episode_start(self, episode: int):
        """Update state for episode start."""
        if self.training_state:
            self.training_state.episode = episode
            self.training_state.current_episode_reward = 0.0
            self.training_state.current_episode_steps = 0
            self.training_state.current_episode_max_x = 0
    
    def update_step(self, step: int, reward: float, mario_x: int, 
                   epsilon: float, learning_rate: float, replay_buffer_size: int):
        """Update state for training step."""
        if not self.training_state:
            return
        
        self.training_state.step = step
        self.training_state.total_steps += 1
        self.training_state.current_episode_reward += reward
        self.training_state.current_episode_steps += 1
        self.training_state.current_episode_max_x = max(self.training_state.current_episode_max_x, mario_x)
        self.training_state.epsilon = epsilon
        self.training_state.learning_rate = learning_rate
        self.training_state.replay_buffer_size = replay_buffer_size
        
        # Auto-save periodically
        if self.training_state.total_steps - self.last_auto_save >= self.auto_save_interval:
            self.save_training_state()
            self.last_auto_save = self.training_state.total_steps
    
    def update_episode_end(self, total_reward: float, max_distance: int, 
                          duration: float, completed: bool):
        """Update state for episode end."""
        if not self.training_state:
            return
        
        self.training_state.total_episodes_completed += 1
        
        if completed:
            self.training_state.successful_episodes += 1
        
        # Update best performance
        if total_reward > self.training_state.best_episode_reward:
            self.training_state.best_episode_reward = total_reward
        
        if max_distance > self.training_state.best_episode_distance:
            self.training_state.best_episode_distance = max_distance
        
        # Add to metrics
        self.training_metrics.add_episode_metrics(total_reward, max_distance, duration, completed)
        
        # Update running averages
        n = self.training_state.total_episodes_completed
        self.training_state.avg_episode_reward = (
            (self.training_state.avg_episode_reward * (n - 1) + total_reward) / n
        )
        self.training_state.avg_episode_distance = (
            (self.training_state.avg_episode_distance * (n - 1) + max_distance) / n
        )
        self.training_state.completion_rate = self.training_state.successful_episodes / n
        
        # Update recent performance
        recent_perf = self.training_metrics.get_recent_performance(100)
        self.training_state.recent_performance = [recent_perf.get('completion_rate', 0.0)]
        
        self.save_training_state()
    
    def create_checkpoint(self, 
                         model_state: Dict[str, Any],
                         optimizer_state: Dict[str, Any],
                         additional_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Create training checkpoint.
        
        Args:
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary
            additional_data: Additional data to save
            
        Returns:
            Path to checkpoint file
        """
        if not self.training_state:
            raise ValueError("No training state available for checkpoint")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_ep{self.training_state.episode}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint_data = {
            'training_state': self.training_state.to_dict(),
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'training_metrics': {
                'episode_rewards': list(self.training_metrics.episode_rewards),
                'episode_distances': list(self.training_metrics.episode_distances),
                'episode_durations': list(self.training_metrics.episode_durations),
                'episode_completions': list(self.training_metrics.episode_completions),
                'loss_history': list(self.training_metrics.loss_history)[-1000:],  # Keep last 1000
                'q_value_history': list(self.training_metrics.q_value_history)[-1000:],
                'epsilon_history': list(self.training_metrics.epsilon_history)[-1000:]
            },
            'timestamp': timestamp,
            'pytorch_version': torch.__version__
        }
        
        if additional_data:
            checkpoint_data['additional_data'] = additional_data
        
        try:
            torch.save(checkpoint_data, checkpoint_path)
            self.logger.info(f"Created checkpoint: {checkpoint_path}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Tuple of (model_state, optimizer_state, metadata)
        """
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Restore training state
            if 'training_state' in checkpoint_data:
                self.training_state = TrainingState.from_dict(checkpoint_data['training_state'])
            
            # Restore training metrics
            if 'training_metrics' in checkpoint_data:
                metrics_data = checkpoint_data['training_metrics']
                self.training_metrics = TrainingMetrics(
                    episode_rewards=deque(metrics_data.get('episode_rewards', []), maxlen=1000),
                    episode_distances=deque(metrics_data.get('episode_distances', []), maxlen=1000),
                    episode_durations=deque(metrics_data.get('episode_durations', []), maxlen=1000),
                    episode_completions=deque(metrics_data.get('episode_completions', []), maxlen=1000),
                    loss_history=deque(metrics_data.get('loss_history', []), maxlen=10000),
                    q_value_history=deque(metrics_data.get('q_value_history', []), maxlen=10000),
                    epsilon_history=deque(metrics_data.get('epsilon_history', []), maxlen=10000),
                    system_metrics=deque(maxlen=1000)
                )
            
            self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
            
            return (
                checkpoint_data.get('model_state_dict', {}),
                checkpoint_data.get('optimizer_state_dict', {}),
                {
                    'timestamp': checkpoint_data.get('timestamp'),
                    'pytorch_version': checkpoint_data.get('pytorch_version'),
                    'additional_data': checkpoint_data.get('additional_data', {})
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def _cleanup_old_checkpoints(self, keep_count: int = 5):
        """Clean up old checkpoint files."""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
            
            if len(checkpoint_files) > keep_count:
                # Sort by modification time (newest first)
                checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                # Remove old checkpoints
                for old_checkpoint in checkpoint_files[keep_count:]:
                    old_checkpoint.unlink()
                    self.logger.debug(f"Removed old checkpoint: {old_checkpoint}")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup old checkpoints: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.training_state:
            return {}
        
        recent_perf = self.training_metrics.get_recent_performance(100)
        
        return {
            'session_info': {
                'session_id': self.training_state.session_id,
                'start_time': datetime.fromtimestamp(self.training_state.start_time).isoformat(),
                'duration_hours': (time.time() - self.training_state.start_time) / 3600,
                'current_episode': self.training_state.episode,
                'total_steps': self.training_state.total_steps
            },
            'progress': {
                'episodes_completed': self.training_state.total_episodes_completed,
                'successful_episodes': self.training_state.successful_episodes,
                'overall_completion_rate': self.training_state.completion_rate,
                'recent_completion_rate': recent_perf.get('completion_rate', 0.0),
                'curriculum_phase': self.training_state.curriculum_phase,
                'training_phase': self.training_state.training_phase
            },
            'performance': {
                'best_episode_reward': self.training_state.best_episode_reward,
                'best_episode_distance': self.training_state.best_episode_distance,
                'avg_episode_reward': self.training_state.avg_episode_reward,
                'avg_episode_distance': self.training_state.avg_episode_distance,
                'recent_avg_reward': recent_perf.get('avg_reward', 0.0),
                'recent_max_distance': recent_perf.get('max_distance', 0),
                'improvement_trend': recent_perf.get('improvement_trend', 0.0)
            },
            'current_state': {
                'epsilon': self.training_state.epsilon,
                'learning_rate': self.training_state.learning_rate,
                'replay_buffer_size': self.training_state.replay_buffer_size,
                'current_episode_reward': self.training_state.current_episode_reward,
                'current_episode_steps': self.training_state.current_episode_steps,
                'current_episode_max_x': self.training_state.current_episode_max_x
            }
        }
    
    def export_training_data(self, output_path: str):
        """Export training data for analysis."""
        data = {
            'training_state': self.training_state.to_dict() if self.training_state else {},
            'training_metrics': {
                'episode_rewards': list(self.training_metrics.episode_rewards),
                'episode_distances': list(self.training_metrics.episode_distances),
                'episode_durations': list(self.training_metrics.episode_durations),
                'episode_completions': list(self.training_metrics.episode_completions),
                'loss_history': list(self.training_metrics.loss_history),
                'q_value_history': list(self.training_metrics.q_value_history),
                'epsilon_history': list(self.training_metrics.epsilon_history)
            },
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Exported training data to: {output_path}")


class SystemHealthMonitor:
    """
    Monitors system health during training.
    """
    
    def __init__(self, warning_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize system health monitor.
        
        Args:
            warning_thresholds: Warning thresholds for system metrics
        """
        self.warning_thresholds = warning_thresholds or {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'gpu_memory_percent': 90.0,
            'gpu_temperature': 85.0,
            'disk_usage_percent': 90.0
        }
        
        self.process = psutil.Process()
        self.gpu_available = len(GPUtil.getGPUs()) > 0
        
        # Health history
        self.health_history = deque(maxlen=100)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Check current system health.
        
        Returns:
            System health metrics and warnings
        """
        health_data = {
            'timestamp': time.time(),
            'cpu_percent': self.process.cpu_percent(),
            'memory_info': self.process.memory_info(),
            'warnings': [],
            'status': 'healthy'
        }
        
        try:
            # Memory metrics
            memory_info = self.process.memory_info()
            system_memory = psutil.virtual_memory()
            
            health_data['memory_usage_mb'] = memory_info.rss / 1024 / 1024
            health_data['memory_percent'] = (memory_info.rss / system_memory.total) * 100
            
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            health_data['disk_usage_percent'] = (disk_usage.used / disk_usage.total) * 100
            
            # GPU metrics
            if self.gpu_available:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        health_data['gpu_memory_used'] = gpu.memoryUsed
                        health_data['gpu_memory_total'] = gpu.memoryTotal
                        health_data['gpu_memory_percent'] = (gpu.memoryUsed / gpu.memoryTotal) * 100
                        health_data['gpu_temperature'] = gpu.temperature
                        health_data['gpu_utilization'] = gpu.load * 100
                except Exception:
                    health_data['gpu_available'] = False
            
            # Check thresholds and generate warnings
            self._check_thresholds(health_data)
            
            # Add to history
            self.health_history.append(health_data)
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            health_data['status'] = 'error'
            health_data['error'] = str(e)
        
        return health_data
    
    def _check_thresholds(self, health_data: Dict[str, Any]):
        """Check health metrics against thresholds."""
        warnings = []
        
        # CPU check
        if health_data['cpu_percent'] > self.warning_thresholds['cpu_percent']:
            warnings.append(f"High CPU usage: {health_data['cpu_percent']:.1f}%")
        
        # Memory check
        if health_data.get('memory_percent', 0) > self.warning_thresholds['memory_percent']:
            warnings.append(f"High memory usage: {health_data['memory_percent']:.1f}%")
        
        # GPU checks
        if health_data.get('gpu_memory_percent', 0) > self.warning_thresholds['gpu_memory_percent']:
            warnings.append(f"High GPU memory usage: {health_data['gpu_memory_percent']:.1f}%")
        
        if health_data.get('gpu_temperature', 0) > self.warning_thresholds['gpu_temperature']:
            warnings.append(f"High GPU temperature: {health_data['gpu_temperature']:.1f}°C")
        
        # Disk check
        if health_data.get('disk_usage_percent', 0) > self.warning_thresholds['disk_usage_percent']:
            warnings.append(f"High disk usage: {health_data['disk_usage_percent']:.1f}%")
        
        health_data['warnings'] = warnings
        if warnings:
            health_data['status'] = 'warning'
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary over recent history."""
        if not self.health_history:
            return {'status': 'no_data'}
        
        recent_data = list(self.health_history)[-10:]  # Last 10 checks
        
        return {
            'current_status': recent_data[-1]['status'],
            'avg_cpu_percent': np.mean([d['cpu_percent'] for d in recent_data]),
            'avg_memory_percent': np.mean([d.get('memory_percent', 0) for d in recent_data]),
            'avg_gpu_temperature': np.mean([d.get('gpu_temperature', 0) for d in recent_data if d.get('gpu_temperature', 0) > 0]),
            'total_warnings': sum(len(d['warnings']) for d in recent_data),
            'recent_warnings': recent_data[-1]['warnings']
        }


if __name__ == "__main__":
    # Test training utilities
    state_manager = TrainingStateManager("test_checkpoints")
    
    # Initialize training state
    config = {'epsilon_start': 1.0, 'learning_rate': 0.00025}
    state = state_manager.initialize_training_state("test_session", config)
    print(f"Initialized state: {state.session_id}")
    
    # Update some metrics
    state_manager.update_episode_start(1)
    state_manager.update_step(1, 1.0, 100, 0.99, 0.00025, 50)
    state_manager.update_episode_end(150.0, 500, 30.5, False)
    
    # Get summary
    summary = state_manager.get_training_summary()
    print(f"Training summary: {summary}")
    
    # Test system health monitor
    health_monitor = SystemHealthMonitor()
    health = health_monitor.check_system_health()
    print(f"System health: {health['status']}")
    
    print("Training utilities test completed!")