"""
CSV Logger for Super Mario Bros AI Training System

Implements comprehensive CSV logging system with multiple log types:
- Training metrics (step-by-step)
- Episode summaries
- Performance monitoring
- Synchronization quality
- Debug events

Based on the CSV logging format specification in docs/csv-logging-format.md
"""

import csv
import json
import logging
import os
import psutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from threading import Lock
import GPUtil


@dataclass
class TrainingLogEntry:
    """Training log entry structure."""
    timestamp: str
    episode: int
    step: int
    reward: float
    total_reward: float
    epsilon: float
    loss: float
    q_value_mean: float
    q_value_std: float
    mario_x: int
    mario_y: int
    mario_x_max: int
    action_taken: int
    processing_time_ms: float
    learning_rate: float
    replay_buffer_size: int


@dataclass
class EpisodeLogEntry:
    """Episode summary log entry structure."""
    timestamp: str
    episode: int
    duration_seconds: float
    total_steps: int
    total_reward: float
    mario_x_final: int
    mario_x_max: int
    level_completed: bool
    death_cause: str
    lives_remaining: int
    score: int
    coins_collected: int
    enemies_killed: int
    powerups_collected: int
    time_remaining: int
    completion_percentage: float
    average_reward_per_step: float
    max_q_value: float
    min_q_value: float
    exploration_actions: int
    exploitation_actions: int


@dataclass
class PerformanceLogEntry:
    """Performance metrics log entry structure."""
    timestamp: str
    episode: int
    step: int
    fps: float
    memory_usage_mb: float
    gpu_memory_mb: float
    gpu_utilization_percent: float
    cpu_percent: float
    network_inference_ms: float
    frame_processing_ms: float
    websocket_latency_ms: float
    disk_io_mb_per_sec: float
    temperature_gpu_celsius: float
    power_draw_watts: float


@dataclass
class SyncQualityLogEntry:
    """Synchronization quality log entry structure."""
    timestamp: str
    episode: int
    step: int
    frame_id: int
    sync_delay_ms: float
    desync_detected: bool
    recovery_time_ms: float
    frame_drops: int
    buffer_size: int
    lua_timestamp: int
    python_timestamp: int
    clock_drift_ms: float


@dataclass
class DebugEventLogEntry:
    """Debug event log entry structure."""
    timestamp: str
    episode: int
    step: int
    event_type: str
    severity: str
    component: str
    message: str
    mario_x: int
    mario_y: int
    action_taken: int
    game_state: str
    stack_trace: str


class CSVLogger:
    """
    Comprehensive CSV logging system for training analysis.
    
    Manages multiple CSV files for different types of logging data:
    - Training metrics (every step)
    - Episode summaries (end of each episode)
    - Performance monitoring (every 100 steps)
    - Synchronization quality (every frame)
    - Debug events (as they occur)
    """
    
    def __init__(self, log_directory: str = "logs", session_id: Optional[str] = None):
        """
        Initialize CSV logger.
        
        Args:
            log_directory: Directory for log files
            session_id: Optional session identifier for file naming
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id
        
        # File paths
        self.training_log_path = self.log_directory / f"training_{session_id}.csv"
        self.episode_log_path = self.log_directory / f"episodes_{session_id}.csv"
        self.performance_log_path = self.log_directory / f"performance_{session_id}.csv"
        self.sync_log_path = self.log_directory / f"sync_quality_{session_id}.csv"
        self.debug_log_path = self.log_directory / f"debug_events_{session_id}.csv"
        
        # Thread safety
        self.lock = Lock()
        
        # Performance monitoring
        self.last_performance_log = 0
        self.performance_log_interval = 100  # Log every 100 steps
        
        # System monitoring
        self.process = psutil.Process()
        self.gpu_available = len(GPUtil.getGPUs()) > 0
        
        # Initialize CSV files
        self._initialize_csv_files()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"CSV Logger initialized with session ID: {session_id}")
    
    def _initialize_csv_files(self):
        """Initialize all CSV files with headers."""
        # Training log headers
        training_headers = [
            'timestamp', 'episode', 'step', 'reward', 'total_reward', 'epsilon',
            'loss', 'q_value_mean', 'q_value_std', 'mario_x', 'mario_y', 'mario_x_max',
            'action_taken', 'processing_time_ms', 'learning_rate', 'replay_buffer_size'
        ]
        self._write_headers(self.training_log_path, training_headers)
        
        # Episode log headers
        episode_headers = [
            'timestamp', 'episode', 'duration_seconds', 'total_steps', 'total_reward',
            'mario_x_final', 'mario_x_max', 'level_completed', 'death_cause',
            'lives_remaining', 'score', 'coins_collected', 'enemies_killed',
            'powerups_collected', 'time_remaining', 'completion_percentage',
            'average_reward_per_step', 'max_q_value', 'min_q_value',
            'exploration_actions', 'exploitation_actions'
        ]
        self._write_headers(self.episode_log_path, episode_headers)
        
        # Performance log headers
        performance_headers = [
            'timestamp', 'episode', 'step', 'fps', 'memory_usage_mb', 'gpu_memory_mb',
            'gpu_utilization_percent', 'cpu_percent', 'network_inference_ms',
            'frame_processing_ms', 'websocket_latency_ms', 'disk_io_mb_per_sec',
            'temperature_gpu_celsius', 'power_draw_watts'
        ]
        self._write_headers(self.performance_log_path, performance_headers)
        
        # Sync quality log headers
        sync_headers = [
            'timestamp', 'episode', 'step', 'frame_id', 'sync_delay_ms',
            'desync_detected', 'recovery_time_ms', 'frame_drops', 'buffer_size',
            'lua_timestamp', 'python_timestamp', 'clock_drift_ms'
        ]
        self._write_headers(self.sync_log_path, sync_headers)
        
        # Debug event log headers
        debug_headers = [
            'timestamp', 'episode', 'step', 'event_type', 'severity', 'component',
            'message', 'mario_x', 'mario_y', 'action_taken', 'game_state', 'stack_trace'
        ]
        self._write_headers(self.debug_log_path, debug_headers)
    
    def _write_headers(self, file_path: Path, headers: List[str]):
        """Write CSV headers to file."""
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_training_step(self, 
                         episode: int,
                         step: int,
                         reward: float,
                         total_reward: float,
                         epsilon: float,
                         loss: float,
                         q_values: Dict[str, float],
                         mario_state: Dict[str, int],
                         action_taken: int,
                         processing_time_ms: float,
                         learning_rate: float,
                         replay_buffer_size: int):
        """
        Log training step data.
        
        Args:
            episode: Current episode number
            step: Current step number
            reward: Step reward
            total_reward: Cumulative episode reward
            epsilon: Current exploration rate
            loss: Neural network loss
            q_values: Q-value statistics (mean, std)
            mario_state: Mario position data (x, y, x_max)
            action_taken: Action ID executed
            processing_time_ms: Step processing time
            learning_rate: Current learning rate
            replay_buffer_size: Current buffer size
        """
        entry = TrainingLogEntry(
            timestamp=datetime.now().isoformat(),
            episode=episode,
            step=step,
            reward=reward,
            total_reward=total_reward,
            epsilon=epsilon,
            loss=loss,
            q_value_mean=q_values.get('mean', 0.0),
            q_value_std=q_values.get('std', 0.0),
            mario_x=mario_state.get('x', 0),
            mario_y=mario_state.get('y', 0),
            mario_x_max=mario_state.get('x_max', 0),
            action_taken=action_taken,
            processing_time_ms=processing_time_ms,
            learning_rate=learning_rate,
            replay_buffer_size=replay_buffer_size
        )
        
        self._write_entry(self.training_log_path, entry)
        
        # Log performance metrics periodically
        if step - self.last_performance_log >= self.performance_log_interval:
            self.log_performance_metrics(episode, step)
            self.last_performance_log = step
    
    def log_episode_summary(self,
                           episode: int,
                           duration_seconds: float,
                           total_steps: int,
                           total_reward: float,
                           mario_final_state: Dict[str, int],
                           level_completed: bool,
                           death_cause: str,
                           game_stats: Dict[str, int],
                           q_value_stats: Dict[str, float],
                           action_stats: Dict[str, int]):
        """
        Log episode summary data.
        
        Args:
            episode: Episode number
            duration_seconds: Episode duration
            total_steps: Total steps in episode
            total_reward: Total episode reward
            mario_final_state: Final Mario state
            level_completed: Whether level was completed
            death_cause: Cause of death if applicable
            game_stats: Game statistics (score, coins, etc.)
            q_value_stats: Q-value statistics
            action_stats: Action statistics
        """
        completion_percentage = min(100.0, (mario_final_state.get('x_max', 0) / 3168.0) * 100.0)
        avg_reward_per_step = total_reward / max(1, total_steps)
        
        entry = EpisodeLogEntry(
            timestamp=datetime.now().isoformat(),
            episode=episode,
            duration_seconds=duration_seconds,
            total_steps=total_steps,
            total_reward=total_reward,
            mario_x_final=mario_final_state.get('x', 0),
            mario_x_max=mario_final_state.get('x_max', 0),
            level_completed=level_completed,
            death_cause=death_cause,
            lives_remaining=game_stats.get('lives', 0),
            score=game_stats.get('score', 0),
            coins_collected=game_stats.get('coins', 0),
            enemies_killed=game_stats.get('enemies_killed', 0),
            powerups_collected=game_stats.get('powerups', 0),
            time_remaining=game_stats.get('time_remaining', 0),
            completion_percentage=completion_percentage,
            average_reward_per_step=avg_reward_per_step,
            max_q_value=q_value_stats.get('max', 0.0),
            min_q_value=q_value_stats.get('min', 0.0),
            exploration_actions=action_stats.get('exploration', 0),
            exploitation_actions=action_stats.get('exploitation', 0)
        )
        
        self._write_entry(self.episode_log_path, entry)
    
    def log_performance_metrics(self, episode: int, step: int):
        """
        Log system performance metrics.
        
        Args:
            episode: Current episode
            step: Current step
        """
        try:
            # System metrics
            memory_info = self.process.memory_info()
            memory_usage_mb = memory_info.rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()
            
            # GPU metrics
            gpu_memory_mb = 0.0
            gpu_utilization = 0.0
            gpu_temperature = 0.0
            gpu_power = 0.0
            
            if self.gpu_available:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        gpu_memory_mb = gpu.memoryUsed
                        gpu_utilization = gpu.load * 100
                        gpu_temperature = gpu.temperature
                        gpu_power = getattr(gpu, 'powerDraw', 0.0)
                except Exception:
                    pass  # GPU metrics not available
            
            # Disk I/O (simplified)
            disk_io = psutil.disk_io_counters()
            disk_io_mb_per_sec = 0.0
            if disk_io:
                # This is a simplified calculation
                disk_io_mb_per_sec = (disk_io.read_bytes + disk_io.write_bytes) / 1024 / 1024
            
            entry = PerformanceLogEntry(
                timestamp=datetime.now().isoformat(),
                episode=episode,
                step=step,
                fps=60.0,  # Target FPS, actual FPS would come from frame capture
                memory_usage_mb=memory_usage_mb,
                gpu_memory_mb=gpu_memory_mb,
                gpu_utilization_percent=gpu_utilization,
                cpu_percent=cpu_percent,
                network_inference_ms=0.0,  # Would be provided by trainer
                frame_processing_ms=0.0,   # Would be provided by trainer
                websocket_latency_ms=0.0,  # Would be provided by websocket server
                disk_io_mb_per_sec=disk_io_mb_per_sec,
                temperature_gpu_celsius=gpu_temperature,
                power_draw_watts=gpu_power
            )
            
            self._write_entry(self.performance_log_path, entry)
            
        except Exception as e:
            self.logger.error(f"Failed to log performance metrics: {e}")
    
    def log_sync_quality(self,
                        episode: int,
                        step: int,
                        frame_id: int,
                        sync_delay_ms: float,
                        desync_detected: bool = False,
                        recovery_time_ms: float = 0.0,
                        frame_drops: int = 0,
                        buffer_size: int = 0,
                        lua_timestamp: int = 0,
                        python_timestamp: int = 0):
        """
        Log synchronization quality metrics.
        
        Args:
            episode: Current episode
            step: Current step
            frame_id: Frame identifier
            sync_delay_ms: Synchronization delay
            desync_detected: Whether desync was detected
            recovery_time_ms: Time to recover from desync
            frame_drops: Number of dropped frames
            buffer_size: Sync buffer size
            lua_timestamp: Lua-side timestamp
            python_timestamp: Python-side timestamp
        """
        clock_drift_ms = python_timestamp - lua_timestamp if lua_timestamp > 0 else 0.0
        
        entry = SyncQualityLogEntry(
            timestamp=datetime.now().isoformat(),
            episode=episode,
            step=step,
            frame_id=frame_id,
            sync_delay_ms=sync_delay_ms,
            desync_detected=desync_detected,
            recovery_time_ms=recovery_time_ms,
            frame_drops=frame_drops,
            buffer_size=buffer_size,
            lua_timestamp=lua_timestamp,
            python_timestamp=python_timestamp,
            clock_drift_ms=clock_drift_ms
        )
        
        self._write_entry(self.sync_log_path, entry)
    
    def log_debug_event(self,
                       episode: int,
                       step: int,
                       event_type: str,
                       severity: str,
                       component: str,
                       message: str,
                       mario_state: Optional[Dict[str, int]] = None,
                       action_taken: int = -1,
                       game_state: Optional[Dict[str, Any]] = None,
                       exception: Optional[Exception] = None):
        """
        Log debug event.
        
        Args:
            episode: Current episode
            step: Current step
            event_type: Event type (error, warning, info, debug)
            severity: Severity level (critical, high, medium, low)
            component: System component
            message: Event description
            mario_state: Mario position data
            action_taken: Last action taken
            game_state: Game state summary
            exception: Exception object if applicable
        """
        mario_x = mario_state.get('x', 0) if mario_state else 0
        mario_y = mario_state.get('y', 0) if mario_state else 0
        
        game_state_json = ""
        if game_state:
            try:
                game_state_json = json.dumps(game_state, separators=(',', ':'))[:500]
            except Exception:
                game_state_json = str(game_state)[:500]
        
        stack_trace = ""
        if exception:
            stack_trace = ''.join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))[:2000]
        
        entry = DebugEventLogEntry(
            timestamp=datetime.now().isoformat(),
            episode=episode,
            step=step,
            event_type=event_type,
            severity=severity,
            component=component,
            message=message[:500],  # Limit message length
            mario_x=mario_x,
            mario_y=mario_y,
            action_taken=action_taken,
            game_state=game_state_json,
            stack_trace=stack_trace
        )
        
        self._write_entry(self.debug_log_path, entry)
    
    def _write_entry(self, file_path: Path, entry: Union[TrainingLogEntry, EpisodeLogEntry, 
                                                        PerformanceLogEntry, SyncQualityLogEntry, 
                                                        DebugEventLogEntry]):
        """
        Write entry to CSV file.
        
        Args:
            file_path: Path to CSV file
            entry: Data entry to write
        """
        with self.lock:
            try:
                with open(file_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if hasattr(entry, '__dict__'):
                        writer.writerow(entry.__dict__.values())
                    else:
                        writer.writerow(entry)
            except Exception as e:
                self.logger.error(f"Failed to write to {file_path}: {e}")
    
    def get_log_files(self) -> Dict[str, Path]:
        """Get paths to all log files."""
        return {
            'training': self.training_log_path,
            'episodes': self.episode_log_path,
            'performance': self.performance_log_path,
            'sync_quality': self.sync_log_path,
            'debug_events': self.debug_log_path
        }
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information."""
        return {
            'session_id': self.session_id,
            'log_directory': str(self.log_directory),
            'log_files': {k: str(v) for k, v in self.get_log_files().items()},
            'gpu_available': self.gpu_available
        }
    
    def close(self):
        """Close logger and flush any pending writes."""
        self.logger.info(f"CSV Logger session {self.session_id} closed")


if __name__ == "__main__":
    # Test CSV logger
    logger = CSVLogger("test_logs")
    
    # Test training step logging
    logger.log_training_step(
        episode=1, step=1, reward=1.0, total_reward=1.0, epsilon=1.0,
        loss=0.5, q_values={'mean': 0.1, 'std': 0.05},
        mario_state={'x': 32, 'y': 208, 'x_max': 32},
        action_taken=1, processing_time_ms=15.2,
        learning_rate=0.00025, replay_buffer_size=100
    )
    
    # Test episode summary logging
    logger.log_episode_summary(
        episode=1, duration_seconds=30.5, total_steps=100, total_reward=150.0,
        mario_final_state={'x': 500, 'x_max': 500},
        level_completed=False, death_cause="enemy_contact",
        game_stats={'lives': 2, 'score': 1200, 'coins': 5, 'enemies_killed': 3, 'powerups': 1, 'time_remaining': 350},
        q_value_stats={'max': 2.5, 'min': -1.2},
        action_stats={'exploration': 80, 'exploitation': 20}
    )
    
    # Test debug event logging
    logger.log_debug_event(
        episode=1, step=50, event_type="warning", severity="medium",
        component="trainer", message="High loss detected",
        mario_state={'x': 250, 'y': 208}, action_taken=2
    )
    
    print("CSV Logger test completed!")
    print(f"Session info: {logger.get_session_info()}")