"""
Main Trainer for Super Mario Bros AI Training System

Orchestrates the complete training process by integrating:
- DQN Agent with experience replay
- WebSocket communication with FCEUX
- Frame capture and preprocessing
- Reward calculation and episode management
- CSV logging and performance monitoring
- Curriculum learning and training phases
"""

import asyncio
import logging
import signal
import time
import torch
import numpy as np
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import threading

# TensorBoard -- optional but recommended
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Import all required components
from python.agents.dqn_agent import DQNAgent
from python.communication.websocket_server import WebSocketServer
from python.capture.frame_capture import FrameCapture
from python.environment.reward_calculator import RewardCalculator
from python.environment.episode_manager import EpisodeManager
from python.mario_logging.csv_logger import CSVLogger
from python.mario_logging.plotter import PerformancePlotter
from python.training.training_utils import TrainingStateManager, SystemHealthMonitor
from python.utils.config_loader import ConfigLoader


class TrainingPhase(Enum):
    """Training phase enumeration."""
    WARMUP = "warmup"
    TRAINING = "training"
    EVALUATION = "evaluation"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class TrainingConfig:
    """Training configuration data structure."""
    max_episodes: int
    max_steps_per_episode: int
    warmup_episodes: int
    save_frequency: int
    evaluation_frequency: int
    target_fps: float
    frame_stack_size: int
    frame_skip: int
    enable_curriculum: bool
    enable_plotting: bool
    reward_clip: float  # Clip rewards to [-reward_clip, +reward_clip]


class MarioTrainer:
    """
    Main trainer class that orchestrates the complete AI training process.
    
    Integrates all subsystems and manages the training loop with proper
    synchronization, error handling, and performance monitoring.
    """
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        """
        Initialize Mario trainer.
        
        Args:
            config_path: Path to training configuration file
        """
        # Load configuration
        self.config_loader = ConfigLoader()
        # Extract just the filename if a full path is provided
        if "/" in config_path or "\\" in config_path:
            config_filename = config_path.split("/")[-1].split("\\")[-1]
        else:
            config_filename = config_path
        self.config = self.config_loader.load_config(config_filename)
        
        # Training state
        self.training_phase = TrainingPhase.WARMUP
        self.is_running = False
        self.should_stop = False
        self.current_episode = 0
        self.current_step = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Performance tracking
        self.frame_times = []
        self.processing_times = []
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        
        # All-time session max fitness -- tracks the furthest Mario has ever reached
        self._session_max_x = 0
        
        # Quality gate state -- only train on episodes above adaptive fitness threshold
        self._quality_gate_enabled = True
        self._qualifying_distances: List[int] = []  # rolling window of qualifying max_x
        self._quality_gate_config: Dict[str, Any] = {}  # loaded from config in _initialize_subsystems
        self._qualifying_episode_count = 0
        self._filtered_episode_count = 0
        self._episode_transitions: List[tuple] = []  # deferred buffer for current episode
        self._episode_did_train = False  # whether this episode had training steps
        
        # Action distribution tracking per episode
        self._episode_action_counts: Counter = Counter()
        
        # Error throttling -- prevents runaway logging when an error repeats
        self._error_counts: Dict[str, int] = {}   # error_key -> count since last log
        self._error_last_logged: Dict[str, float] = {}  # error_key -> timestamp
        self._error_throttle_interval = 30.0  # seconds between repeated error logs
        self._error_throttle_burst = 3  # log first N occurrences immediately
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing Mario Trainer - Session: {self.session_id}")
        
        # Initialize all subsystems
        self._initialize_subsystems()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        self.logger.info("Mario Trainer initialization completed")
    
    def _initialize_subsystems(self):
        """Initialize all training subsystems."""
        try:
            # Training configuration
            training_config = self.config.get('training', {})
            self.training_config = TrainingConfig(
                max_episodes=training_config.get('max_episodes', 50000),
                max_steps_per_episode=training_config.get('max_steps_per_episode', 4500),
                warmup_episodes=training_config.get('warmup_episodes', 50),
                save_frequency=training_config.get('save_frequency', 500),
                evaluation_frequency=training_config.get('evaluation_frequency', 500),
                target_fps=30,
                frame_stack_size=4,
                frame_skip=training_config.get('frame_skip', 4),
                enable_curriculum=training_config.get('curriculum', {}).get('enabled', True),
                enable_plotting=False,
                reward_clip=10.0  # Clip rewards to [-10, +10]; was 1.0 which flattened
                                  # all forward-motion steps to the same value and made
                                  # death penalties negligible, causing policy collapse
            )
            
            # TensorBoard writer -- logs scalars for loss, reward, Q-values, epsilon, actions
            self.tb_writer: Optional['SummaryWriter'] = None
            if TENSORBOARD_AVAILABLE:
                tb_log_dir = Path("runs") / self.session_id
                self.tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
                self.logger.info(f"TensorBoard logging enabled -> {tb_log_dir}")
            else:
                self.logger.warning(
                    "TensorBoard not available (pip install tensorboard). "
                    "Training will proceed without TB logging."
                )
            
            # Frame skip counter (acts every N frames, repeats last action for skipped frames)
            self._frame_skip_counter = 0
            self._last_action_id = 0
            self._last_action_buttons = {}
            
            # Initialize CSV logger
            self.csv_logger = CSVLogger(
                log_directory="logs",
                session_id=self.session_id
            )
            
            # Initialize training state manager
            self.state_manager = TrainingStateManager(
                checkpoint_dir="checkpoints",
                auto_save_interval=100
            )
            
            # Initialize system health monitor
            self.health_monitor = SystemHealthMonitor()
            
            # Initialize DQN agent
            agent_config = {**self.config.get('training', {}), **self.config.get('performance', {})}
            self.agent = DQNAgent(agent_config)
            
            # Initialize reward calculator with stuck detection config from training section
            reward_config = self.config.get('rewards', {})
            stuck_config = {
                'stuck_timeout_frames': training_config.get('stuck_timeout_frames', 300),
                'stuck_grace_frames': training_config.get('stuck_grace_frames', 60),
                'stuck_penalty_per_frame': training_config.get('stuck_penalty_per_frame', -0.1),
                'stuck_progress_threshold': training_config.get('stuck_progress_threshold', 5),
            }
            self.reward_calculator = RewardCalculator(reward_config, stuck_config=stuck_config)
            
            # Initialize episode manager
            self.episode_manager = EpisodeManager(
                reward_calculator=self.reward_calculator,
                log_directory="logs",
                csv_filename=f"episodes_{self.session_id}.csv"
            )
            
            # Initialize frame capture with light-load defaults from training config
            capture_config = self.config.get('capture', {})
            self.frame_capture = FrameCapture(
                window_title=capture_config.get('window_title', 'FCEUX'),
                target_fps=capture_config.get('target_fps', self.training_config.target_fps),
                frame_stack_size=capture_config.get('frame_stack_size', self.training_config.frame_stack_size),
                target_size=tuple(capture_config.get('target_size', [84, 84]))
            )
            
            # Initialize WebSocket server
            network_config = self.config.get('network', {})
            websocket_config = self.config.get('websocket', {})
            self.websocket_server = WebSocketServer(
                host=websocket_config.get('host') or network_config.get('host', 'localhost'),
                port=websocket_config.get('port') or network_config.get('port', 8765)
            )
            
            # Register WebSocket handlers
            self._register_websocket_handlers()
            
            # Initialize performance plotter (disabled to prevent freezing)
            # Real-time plotting causes matplotlib GUI issues in background threads
            self.plotter = PerformancePlotter(
                log_directory="logs",
                session_id=self.session_id
            ) if self.training_config.enable_plotting else None
            
            # Disable real-time plotting to prevent freezing
            self.enable_realtime_plotting = False
            
            # Initialize training state
            self.training_state = self.state_manager.initialize_training_state(
                self.session_id, 
                self.config.get('training', {})
            )
            
            # Load quality gate configuration
            qg = training_config.get('quality_gate', {})
            self._quality_gate_enabled = qg.get('enabled', True)
            self._quality_gate_config = {
                'percentile': qg.get('percentile', 25),
                'window_size': qg.get('window_size', 100),
                'min_qualifying': qg.get('min_qualifying', 20),
                'warmup_bypass': qg.get('warmup_bypass_episodes', 50),
                'floor_x': qg.get('floor_x', 200),
                'frontier_ratio': qg.get('frontier_ratio', 0.8),
                'periodic_every': qg.get('periodic_qualify_every', 20),
            }
            self.logger.info(f"Quality gate {'enabled' if self._quality_gate_enabled else 'disabled'}: {self._quality_gate_config}")
            
            self.logger.info("All subsystems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize subsystems: {e}")
            raise
    
    def _register_websocket_handlers(self):
        """Register WebSocket message handlers."""
        # Register binary handler as FALLBACK for legacy binary game state data
        self.websocket_server.register_binary_handler(self._handle_binary_game_state)
        
        # Register screen frame handler for Lua gui.gdscreenshot() data
        self.websocket_server.register_screen_frame_handler(self._handle_screen_frame)
        
        # Register JSON handlers
        # PRIMARY: game_state now arrives as JSON (eliminates binary parsing bugs)
        self.websocket_server.register_json_handler('game_state', self._handle_json_game_state)
        self.websocket_server.register_json_handler('episode_event', self._handle_episode_event)
        self.websocket_server.register_json_handler('frame_advance', self._handle_frame_advance)
        self.websocket_server.register_json_handler('error', self._handle_lua_error)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        # Note: Signal handlers are set up in main.py to avoid conflicts
        # This method is kept for compatibility but doesn't set handlers
        self.logger.debug("Signal handlers managed by main.py")
    
    async def start_training(self, resume_from_checkpoint: Optional[str] = None):
        """
        Start the training process.
        
        Args:
            resume_from_checkpoint: Optional checkpoint path to resume from
        """
        try:
            self.logger.info("Starting Mario AI training...")
            
            # Resume from checkpoint if provided
            if resume_from_checkpoint:
                await self._resume_from_checkpoint(resume_from_checkpoint)
            
            # Start WebSocket server
            await self.websocket_server.start_server()
            
            # Wait for client connection with timeout
            self.logger.info("Waiting for FCEUX client connection...")
            connection_timeout = 300  # 5 minutes timeout
            connection_start = time.time()
            
            while not self.websocket_server.is_client_connected() and not self.should_stop:
                await asyncio.sleep(1.0)
                
                # Check for timeout
                if time.time() - connection_start > connection_timeout:
                    self.logger.error("Timeout waiting for FCEUX client connection")
                    self.logger.error("Make sure FCEUX is running with the Lua script loaded!")
                    return
                
                # Log waiting status every 30 seconds
                if int(time.time() - connection_start) % 30 == 0:
                    elapsed = int(time.time() - connection_start)
                    self.logger.info(f"Still waiting for FCEUX connection... ({elapsed}s elapsed)")
            
            if self.should_stop:
                return
            
            self.logger.info("Client connected, starting training loop...")
            
            # Start frame capture from FCEUX window
            try:
                self.frame_capture.start_capture()
                self.logger.info("Frame capture started successfully")
            except Exception as e:
                self.logger.warning(f"Frame capture start failed: {e}")
                self.logger.warning("Training will continue with zero frames - DQN relies on state vector only")
                self.logger.warning("Make sure FCEUX window is visible and not minimized")
            
            # Skip real-time plotting to prevent freezing issues
            # The plotter will still create static analysis at the end
            if self.plotter and self.enable_realtime_plotting:
                self.logger.info("Real-time plotting disabled to prevent freezing issues")
                # plotting_thread = threading.Thread(
                #     target=self.plotter.start_realtime_monitoring,
                #     daemon=True
                # )
                # plotting_thread.start()
            
            # Start main training loop
            self.is_running = True
            await self._training_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self.should_stop = True
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            # Don't re-raise the exception to prevent cleanup from running
            # Instead, set should_stop flag and let training loop handle it gracefully
            self.should_stop = True
        finally:
            # Only cleanup if we're explicitly stopping training AND not due to a recoverable exception
            # This prevents cleanup from running on temporary exceptions that can be recovered from
            if self.should_stop and not self.is_running:
                self.logger.info("Training stopped intentionally, performing cleanup")
                await self._cleanup()
            elif self.should_stop:
                self.logger.info("Training flagged to stop but still running - skipping cleanup to allow recovery")
            else:
                self.logger.info("Training completed normally, skipping cleanup to maintain connection")
    
    async def _training_loop(self):
        """Main training loop."""
        self.logger.info("Entering main training loop...")
        
        try:
            while not self.should_stop and self.current_episode < self.training_config.max_episodes:
                # Check if client is still connected
                if not self.websocket_server.is_client_connected():
                    self.logger.warning("Client disconnected, waiting for reconnection...")
                    connection_wait_start = time.time()
                    connection_timeout = 60.0  # 1 minute timeout for reconnection
                    
                    while not self.websocket_server.is_client_connected() and not self.should_stop:
                        await asyncio.sleep(1.0)
                        
                        # Check for reconnection timeout
                        if time.time() - connection_wait_start > connection_timeout:
                            self.logger.error("Client reconnection timeout - stopping training")
                            self.should_stop = True
                            break
                    
                    if self.should_stop:
                        break
                    
                    self.logger.info("Client reconnected, resuming training...")
                
                # Start new episode
                await self._start_episode()
                
                # Episode loop - wait for game state data to drive the training
                episode_start_time = time.time()
                step_in_episode = 0
                episode_timeout = 60.0  # 60 second timeout per episode
                last_game_state_time = time.time()
                
                self.logger.info(f"Starting episode {self.current_episode + 1}, waiting for game state data...")
                
                while (not self.should_stop and
                       step_in_episode < self.training_config.max_steps_per_episode and
                       self.websocket_server.is_client_connected()):
                    
                    # Check for episode timeout (no game data received)
                    current_time = time.time()
                    if current_time - last_game_state_time > episode_timeout:
                        self.logger.warning(f"Episode timeout - no game data received for {episode_timeout}s")
                        self.logger.warning("This usually means the Lua script is not running or not sending data")
                        break
                    
                    # Check if episode is still running (updated by _handle_game_state)
                    if (self.episode_manager.current_episode and
                        self.episode_manager.current_episode.status.value != "running"):
                        self.logger.info("Episode ended by game state handler")
                        break
                    
                    # Wait for game state data (processed by _handle_game_state)
                    await asyncio.sleep(0.1)
                    
                    # Update last game state time if we received data recently
                    if hasattr(self, '_last_game_state_received'):
                        last_game_state_time = self._last_game_state_received
                    
                    # Update FPS tracking
                    self._update_fps_tracking(current_time)
                    
                    # Check system health periodically
                    if step_in_episode % 100 == 0:
                        health_data = self.health_monitor.check_system_health()
                        if health_data['warnings']:
                            self.logger.warning(f"System health warnings: {health_data['warnings']}")
                    
                    step_in_episode += 1
                
                # End episode
                episode_duration = time.time() - episode_start_time
                await self._end_episode(episode_duration)
                
                # Save checkpoint periodically
                if self.current_episode % self.training_config.save_frequency == 0:
                    await self._save_checkpoint()
                
                # Skip evaluation completely to prevent connection drops
                # Evaluation causes WebSocket disconnection issues
                # if self.current_episode % self.training_config.evaluation_frequency == 0:
                #     await self._run_evaluation()
                
                self.current_episode += 1
            
            self.logger.info("Training loop completed")
            
        except Exception as e:
            self.logger.error(f"Error in training loop: {e}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            # Don't re-raise the exception to prevent cleanup from running
            # Set should_stop flag to gracefully exit the training loop
            self.should_stop = True
    
    async def _start_episode(self):
        """Start a new training episode.
        
        This is the ONLY place that creates episodes in the episode_manager.
        _handle_game_state no longer auto-creates episodes.
        """
        self.logger.info(f"Starting episode {self.current_episode + 1}")
        
        # Update training phase based on episode count
        self._update_training_phase()
        
        # Send training control command to Lua script
        reset_sent = await self.websocket_server.send_training_control(
            command="reset",
            episode_id=self.current_episode + 1,
            reset_to_level="1-1"
        )
        
        if reset_sent:
            await asyncio.sleep(0.5)
            self.logger.debug(f"Reset command sent for episode {self.current_episode + 1}")
        else:
            self.logger.error(f"Failed to send reset for episode {self.current_episode + 1}")
        
        # Create episode in episode manager (single source of truth).
        # IMPORTANT: lives=0 so the first real frame doesn't trigger false
        # death detection (SMB stores displayed_lives - 1 at 0x075A, so
        # "3 lives" on screen = byte value 2, not 3).  Setting to 0 ensures
        # any real value is >= previous and no spurious death fires.
        initial_state = {
            'mario_x': 40,   # World 1-1 start position
            'mario_y': 176,
            'score': 0,
            'lives': 0,      # 0 prevents false death on first frame
            'time': 400
        }
        self.episode_manager.start_episode(initial_state)
        
        # Reset episode-specific state
        self.current_step = 0
        self._frame_skip_counter = 0
        self.frame_times.clear()
        self.processing_times.clear()
        self._episode_action_counts.clear()
        self._episode_transitions.clear()  # Clear deferred transition buffer
        self._episode_did_train = False
        
        # Update state manager
        self.state_manager.update_episode_start(self.current_episode + 1)
    
    async def _end_episode(self, episode_duration: float):
        """End current episode and update statistics."""
        if not self.episode_manager.current_episode:
            return
        
        episode_stats = self.episode_manager.current_episode
        
        # Calculate episode metrics
        total_reward = episode_stats.total_reward
        max_distance = episode_stats.max_x_reached
        completed = episode_stats.level_completed
        
        # End episode in episode manager (always -- for CSV logging)
        episode_data = {
            'final_score': episode_stats.score,
            'time_remaining': episode_stats.time_remaining,
            'coins_collected': episode_stats.coins_collected,
            'lives_used': episode_stats.lives_used,
            'termination_reason': episode_stats.termination_reason
        }
        
        completed_episode = self.episode_manager.end_episode(episode_data)
        
        # --- QUALITY GATE CHECK ---
        episode_qualifies = self._check_quality_gate(max_distance)
        # Always qualify level completions -- never discard a victory
        if completed:
            episode_qualifies = True
        
        if episode_qualifies:
            # COMMIT: Store all buffered transitions in replay buffer
            for transition in self._episode_transitions:
                pf, ps, aid, rew, nf, ns, done = transition
                self.agent.store_experience(
                    pf.to(self.agent.device), ps.to(self.agent.device),
                    aid, rew,
                    nf.to(self.agent.device), ns.to(self.agent.device),
                    done
                )
            
            # Run training steps proportional to transitions committed
            train_count = max(1, len(self._episode_transitions) // 4)  # 1 train per 4 transitions
            for _ in range(train_count):
                if self.training_phase != TrainingPhase.WARMUP:
                    metrics = self.agent.train_step()
                    if metrics and self.tb_writer is not None:
                        gs = self.agent.training_step
                        self.tb_writer.add_scalar("train/loss", metrics.get('loss', 0), gs)
                        self.tb_writer.add_scalar("train/mean_q_value", metrics.get('mean_q_value', 0), gs)
                        self.tb_writer.add_scalar("train/epsilon", metrics.get('epsilon', 0), gs)
            
            # Decay epsilon only for qualifying episodes
            self.agent.episode_end(total_reward, episode_stats.frames_processed)
            
            # Update qualifying episode window
            self._qualifying_distances.append(max_distance)
            qg = self._quality_gate_config
            if len(self._qualifying_distances) > qg.get('window_size', 100):
                self._qualifying_distances.pop(0)
            self._qualifying_episode_count += 1
            
            self.logger.info(f"  [QUALIFIED] ep {self.current_episode + 1}: "
                           f"x={max_distance}, threshold={self._get_quality_threshold()}, "
                           f"transitions={len(self._episode_transitions)}")
        else:
            # DISCARD: Don't store transitions, don't train, don't decay epsilon
            self._filtered_episode_count += 1
            self.logger.info(f"  [FILTERED] ep {self.current_episode + 1}: "
                           f"x={max_distance} < threshold={self._get_quality_threshold()}")
        
        # Clear deferred buffer
        self._episode_transitions.clear()
        
        # Update state manager (always, for stats tracking)
        self.state_manager.update_episode_end(
            total_reward, max_distance, episode_duration, completed
        )
        
        # Log episode summary to CSV (always, even filtered episodes)
        if completed_episode:
            # Get Q-value statistics from agent
            agent_stats = self.agent.get_stats()
            q_value_stats = {
                'max': agent_stats.get('episode_reward_max', 0.0),
                'min': agent_stats.get('episode_reward_min', 0.0)
            }
            
            # Get action statistics from actual counts
            total_actions = sum(self._episode_action_counts.values()) or 1
            action_stats = {
                'exploration': int(episode_stats.frames_processed * self.agent.epsilon),
                'exploitation': int(episode_stats.frames_processed * (1 - self.agent.epsilon))
            }
            
            # --- TensorBoard episode-level logging ---
            if self.tb_writer is not None:
                ep = self.current_episode + 1
                self.tb_writer.add_scalar("episode/reward", total_reward, ep)
                self.tb_writer.add_scalar("episode/distance", max_distance, ep)
                self.tb_writer.add_scalar("episode/duration_s", episode_duration, ep)
                self.tb_writer.add_scalar("episode/epsilon", self.agent.epsilon, ep)
                self.tb_writer.add_scalar("episode/steps", episode_stats.frames_processed, ep)
                
                # Action distribution histogram -- detect collapsed exploration
                if self._episode_action_counts:
                    action_tensor = torch.zeros(12)
                    for aid, cnt in self._episode_action_counts.items():
                        if 0 <= aid < 12:
                            action_tensor[aid] = cnt
                    self.tb_writer.add_histogram("episode/action_distribution", action_tensor, ep)
                    
                    # Scalar: number of unique actions used (low = collapsed)
                    self.tb_writer.add_scalar(
                        "episode/unique_actions", len(self._episode_action_counts), ep
                    )
                    # Scalar: entropy of action distribution (0 = always same action)
                    probs = action_tensor / action_tensor.sum().clamp(min=1)
                    entropy = -(probs * (probs + 1e-8).log()).sum().item()
                    self.tb_writer.add_scalar("episode/action_entropy", entropy, ep)
            
            self.csv_logger.log_episode_summary(
                episode=self.current_episode + 1,
                duration_seconds=episode_duration,
                total_steps=episode_stats.frames_processed,
                total_reward=total_reward,
                mario_final_state={'x': episode_stats.max_x_reached, 'x_max': episode_stats.max_x_reached},
                level_completed=completed,
                death_cause=episode_stats.termination_reason,
                game_stats={
                    'lives': getattr(episode_stats, 'lives_remaining', getattr(episode_stats, 'lives_used', 3)),
                    'score': getattr(episode_stats, 'score', 0),
                    'coins': getattr(episode_stats, 'coins_collected', 0),
                    'enemies_killed': getattr(episode_stats, 'enemies_killed', 0),
                    'powerups': getattr(episode_stats, 'powerups_collected', 0),
                    'time_remaining': getattr(episode_stats, 'time_remaining', 0)
                },
                q_value_stats=q_value_stats,
                action_stats=action_stats
            )
        
        # Update all-time session max fitness
        if max_distance > self._session_max_x:
            self._session_max_x = max_distance
            self.logger.info(
                f"*** NEW SESSION BEST: x={self._session_max_x} "
                f"({self._session_max_x / 3168.0 * 100:.1f}% of 1-1) ***"
            )
        
        # TensorBoard: all-time max fitness
        if self.tb_writer is not None:
            ep = self.current_episode + 1
            self.tb_writer.add_scalar("session/max_x_ever", self._session_max_x, ep)
        
        self.logger.info(
            f"Episode {self.current_episode + 1} completed: "
            f"Reward={total_reward:.1f}, Distance={max_distance}, "
            f"MAX={self._session_max_x}, "
            f"Duration={episode_duration:.1f}s, Completed={completed}"
        )
    
    async def _handle_json_game_state(self, data: Dict[str, Any]):
        """
        Handle game state arriving as JSON (preferred protocol).
        
        The Lua script now sends game state as a flat JSON object with type="game_state".
        This eliminates binary struct packing/unpacking bugs entirely.
        
        Args:
            data: Parsed JSON dictionary from Lua
        """
        frame_id = data.get('frame_id', 0)
        # The JSON dict IS the game state -- no binary parsing needed
        mario_x_vel = data.get('mario_x_vel', 0)
        mario_y_vel = data.get('mario_y_vel', 0)
        game_state = {
            'mario_x': data.get('mario_x', 0),
            'mario_y': data.get('mario_y', 0),
            'mario_x_vel': mario_x_vel,
            'mario_y_vel': mario_y_vel,
            'mario_state': data.get('mario_state', 0),
            'powerup': data.get('powerup', 0),
            'power_state': data.get('powerup', 0),
            'lives': data.get('lives', 3),
            'world': data.get('world', 1),
            'level': data.get('level', 1),
            'time': data.get('time', 400),
            'score': data.get('score', 0),
            'coins': data.get('coins', 0),
            'timestamp': data.get('timestamp', time.time()),
            # Enhanced features for 20-feature state vector (hazard avoidance)
            'on_ground': data.get('on_ground', 0),
            'direction': data.get('direction', 0),
            'invincible': data.get('invincible', 0),
            'closest_enemy_distance': data.get('closest_enemy_dist', 999.0),
            'enemy_count': data.get('threat_count', 0),
            'threats_ahead': data.get('threats_ahead', 0),
            'threats_behind': data.get('threats_behind', 0),
            'pit_detected': data.get('pit_detected', False),
            'solid_tiles_ahead': data.get('solid_tiles_ahead', 0),
            'powerup_present': data.get('powerup_present', False),
            'velocity_magnitude': (mario_x_vel**2 + mario_y_vel**2)**0.5,
            'facing_direction': data.get('direction', 1),
            'is_level_complete': data.get('is_level_complete', False),
            'is_dead': data.get('is_dead', False),
            'level_progress': data.get('level_progress', 0.0),
        }
        await self._process_game_state_dict(frame_id, game_state)
    
    async def _handle_binary_game_state(self, frame_id: int, game_state_data: bytes):
        """
        Handle binary game state data (legacy fallback).
        
        Kept for backward compatibility if Lua sends binary 0x01 packets.
        New Lua code sends JSON instead.
        
        Args:
            frame_id: Frame identifier
            game_state_data: Binary game state data
        """
        game_state = self.frame_capture.parse_game_state(game_state_data)
        await self._process_game_state_dict(frame_id, game_state)
    
    async def _process_game_state_dict(self, frame_id: int, game_state: Dict[str, Any]):
        """
        Core game state processing logic shared by JSON and binary handlers.
        
        Args:
            frame_id: Frame identifier
            game_state: Parsed game state dictionary
        """
        try:
            step_start_time = time.time()
            
            # Track when we last received game state data
            self._last_game_state_received = step_start_time
            
            # Episodes are created exclusively by _start_episode().
            # If no episode exists or it's not running, skip this frame.
            if not self.episode_manager.current_episode:
                self.logger.debug("No active episode, skipping frame (waiting for _start_episode)")
                return
            elif self.episode_manager.current_episode.status.value != "running":
                self.logger.debug(f"Episode not running (status={self.episode_manager.current_episode.status.value}), skipping frame")
                return
            
            # Process frame in episode manager
            frame_reward, reward_components, is_terminal = self.episode_manager.process_frame(
                game_state,
                sync_quality=1.0
            )
            
            # REWARD CLIPPING: clip to [-clip, +clip] for Q-value stability
            clip_val = self.training_config.reward_clip
            frame_reward = max(-clip_val, min(clip_val, frame_reward))
            
            # Log terminal state detection
            if is_terminal:
                self.logger.info(f"Terminal state detected in episode manager: {self.episode_manager.current_episode.termination_reason if self.episode_manager.current_episode else 'unknown'}")
            
            # FRAME SKIPPING: only select a new action every N frames
            # On skipped frames, repeat the previous action and accumulate reward
            self._frame_skip_counter += 1
            select_new_action = (self._frame_skip_counter >= self.training_config.frame_skip) or is_terminal
            
            if not select_new_action and hasattr(self, '_last_action_id'):
                # Repeat previous action on skipped frames
                action_sent = await self.websocket_server.send_action(
                    self._last_action_buttons, frame_id, action_id=self._last_action_id
                )
                return  # Skip network forward pass on skipped frames
            
            # Reset frame skip counter
            self._frame_skip_counter = 0
            
            # Get preprocessed frames and state vector
            frames, state_vector = self.frame_capture.process_frame(game_state)
            
            # Convert to tensors and ensure proper dimensions
            if not isinstance(frames, torch.Tensor):
                frames = torch.from_numpy(frames).float()
            if not isinstance(state_vector, torch.Tensor):
                state_vector = torch.from_numpy(state_vector).float()
            
            # Ensure frames have batch dimension: (1, height, width, channels)
            if len(frames.shape) == 3:
                frames = frames.unsqueeze(0)
            elif len(frames.shape) == 4 and frames.shape[0] != 1:
                frames = frames[:1]
            
            # Convert from channels-last to channels-first format for PyTorch
            if len(frames.shape) == 4:
                frames = frames.permute(0, 3, 1, 2)
            
            # Ensure state vector has batch dimension: (1, features)
            if len(state_vector.shape) == 1:
                state_vector = state_vector.unsqueeze(0)
            elif len(state_vector.shape) == 2 and state_vector.shape[0] != 1:
                state_vector = state_vector[:1]
            
            # Move to device
            frames = frames.to(self.agent.device)
            state_vector = state_vector.to(self.agent.device)
            
            # Shape validation before training steps
            self._validate_tensor_shapes(frames, state_vector)
            
            # Agent action selection -- use uniform random during warmup to
            # fill replay buffer with diverse experience (NoisyNet init bias
            # otherwise causes directional lock before training starts)
            is_warmup = self.training_phase == TrainingPhase.WARMUP
            action_id = self.agent.select_action(
                frames, state_vector, training=True, force_random=is_warmup
            )
            
            # Convert action ID to button mapping
            action_buttons = self._action_id_to_buttons(action_id)
            
            # Track action distribution for this episode
            self._episode_action_counts[action_id] += 1
            
            # Cache for frame skipping
            self._last_action_id = action_id
            self._last_action_buttons = action_buttons
            
            # Send action to Lua script (include action_id for reliable Lua-side lookup)
            action_sent = await self.websocket_server.send_action(action_buttons, frame_id, action_id=action_id)
            
            if action_sent:
                self.logger.debug(f"Sent action {action_id} (buttons: {action_buttons}) for frame {frame_id}")
            else:
                self.logger.warning(f"Failed to send action {action_id} for frame {frame_id} - connection may be lost")
            
            # QUALITY GATE: Buffer transitions instead of storing directly.
            # Transitions are committed or discarded in _end_episode() based on
            # whether the episode meets the adaptive fitness threshold.
            if hasattr(self, 'previous_frames') and hasattr(self, 'previous_state_vector'):
                prev_frames = self.previous_frames
                prev_state = self.previous_state_vector
                
                # Ensure previous frames are in channels-first format
                if hasattr(prev_frames, 'shape') and len(prev_frames.shape) == 4 and prev_frames.shape[-1] == 4:
                    prev_frames = prev_frames.permute(0, 3, 1, 2)
                
                # Buffer the transition -- will be committed or discarded at episode end
                self._episode_transitions.append((
                    prev_frames.cpu(),
                    prev_state.cpu(),
                    self.previous_action_id,
                    frame_reward,
                    frames.cpu(),
                    state_vector.cpu(),
                    is_terminal
                ))
            
            # Training steps are deferred until we know if the episode qualifies.
            # We still need to run train_step() during warmup to fill the buffer
            # and during qualifying episodes (handled in _end_episode).
            # For now, skip per-frame training -- it will be done in batch at episode end.
            training_metrics = {}
            
            # Update state for next step
            self.previous_frames = frames
            self.previous_state_vector = state_vector
            self.previous_action_id = action_id
            
            # Calculate processing time
            processing_time_ms = (time.time() - step_start_time) * 1000
            self.processing_times.append(processing_time_ms)
            
            # Update state manager
            self.state_manager.update_step(
                self.current_step,
                frame_reward,
                game_state.get('mario_x', 0),
                self.agent.epsilon,
                self.agent.learning_rate,
                len(self.agent.replay_buffer)
            )
            
            # Log training step to CSV
            q_values = {
                'mean': training_metrics.get('mean_q_value', 0.0),
                'std': 0.0  # Simplified
            }
            
            mario_state = {
                'x': game_state.get('mario_x', 0),
                'y': game_state.get('mario_y', 0),
                'x_max': self.episode_manager.current_episode.max_x_reached if self.episode_manager.current_episode else 0
            }
            
            self.csv_logger.log_training_step(
                episode=self.current_episode + 1,
                step=self.current_step,
                reward=frame_reward,
                total_reward=self.episode_manager.current_episode.total_reward if self.episode_manager.current_episode else 0,
                epsilon=self.agent.epsilon,
                loss=training_metrics.get('loss', 0.0),
                q_values=q_values,
                mario_state=mario_state,
                action_taken=action_id,
                processing_time_ms=processing_time_ms,
                learning_rate=self.agent.learning_rate,
                replay_buffer_size=len(self.agent.replay_buffer)
            )
            
            # Log sync quality -- throttled to every 100 steps to avoid huge CSVs
            if self.current_step % 100 == 0:
                self.csv_logger.log_sync_quality(
                    episode=self.current_episode + 1,
                    step=self.current_step,
                    frame_id=frame_id,
                    sync_delay_ms=processing_time_ms,
                    desync_detected=False,
                    recovery_time_ms=0.0,
                    frame_drops=0,
                    buffer_size=1,
                    lua_timestamp=int(time.time() * 1000),
                    python_timestamp=int(time.time() * 1000)
                )
            
            self.current_step += 1
            
            # Log periodic status
            if self.current_step % 100 == 0:
                self.logger.info(f"Episode {self.current_episode + 1}, Step {self.current_step}: "
                               f"Reward={frame_reward:.2f}, Total={self.episode_manager.current_episode.total_reward:.2f}, "
                               f"Mario X={game_state.get('mario_x', 0)}, "
                               f"MAX={self._session_max_x}, Action={action_id}")
            
        except Exception as e:
            # Throttled error logging -- prevents flooding logs when an error repeats
            error_key = f"game_state:{type(e).__name__}"
            if self._should_log_error(error_key):
                suppressed = self._error_counts.get(error_key, 0)
                suffix = f" (suppressed {suppressed} duplicates)" if suppressed > 0 else ""
                self.logger.error(f"Error handling game state: {e}{suffix}")
                self.logger.error(f"Exception type: {type(e).__name__}")
                
                # Log debug event to CSV only when we actually log
                self.csv_logger.log_debug_event(
                    episode=self.current_episode + 1,
                    step=self.current_step,
                    event_type="error",
                    severity="high",
                    component="trainer",
                    message=f"Game state processing error: {str(e)}{suffix}",
                    exception=e
                )
    
    def _validate_tensor_shapes(self, frames: torch.Tensor, state_vector: torch.Tensor):
        """
        Validate tensor shapes match network expectations.
        
        Args:
            frames: Frame tensor to validate
            state_vector: State vector tensor to validate
        """
        # Log tensor shapes for debugging
        self.logger.debug(f"Frame tensor shape: {frames.shape}, State vector shape: {state_vector.shape}")
        self.logger.debug(f"Expected frame_stack_size from config: {self.training_config.frame_stack_size}")
        
        # Validate frame tensor shape
        expected_channels = self.training_config.frame_stack_size
        if len(frames.shape) == 4:
            actual_channels = frames.shape[1]  # Channels-first format: (batch, channels, height, width)
        else:
            actual_channels = frames.shape[0] if len(frames.shape) == 3 else 0
        
        if actual_channels != expected_channels:
            # Throttled -- this fires every frame when mismatched, so limit output
            if self._should_log_error("tensor_shape_mismatch"):
                suppressed = self._error_counts.get("tensor_shape_mismatch", 0)
                self.logger.error(
                    f"TENSOR SHAPE MISMATCH: expected {expected_channels} channels, "
                    f"got {actual_channels} (shape={frames.shape})"
                    + (f" [suppressed {suppressed} duplicates]" if suppressed else "")
                )
                self.csv_logger.log_debug_event(
                    episode=self.current_episode + 1,
                    step=self.current_step,
                    event_type="tensor_shape_mismatch",
                    severity="critical",
                    component="trainer",
                    message=f"Frame tensor shape mismatch: expected {expected_channels} channels, got {actual_channels}",
                    exception=None
                )
        
        # Validate state vector shape
        expected_state_size = 20  # Enhanced game state vector size (was 12)
        if len(state_vector.shape) == 2:
            actual_state_size = state_vector.shape[1]  # (batch, features)
        else:
            actual_state_size = state_vector.shape[0] if len(state_vector.shape) == 1 else 0
        
        if actual_state_size != expected_state_size:
            self.logger.warning(f"State vector size mismatch: expected {expected_state_size}, got {actual_state_size}")
    
    def _action_id_to_buttons(self, action_id: int) -> Dict[str, bool]:
        """
        Convert action ID to button mapping.
        
        MUST match the Lua ACTION_MAPPING table exactly:
          [0]  = No action
          [1]  = Right
          [2]  = Left
          [3]  = Jump (A)
          [4]  = Right + Jump
          [5]  = Left + Jump
          [6]  = Run/Fire (B)
          [7]  = Right + Run
          [8]  = Left + Run
          [9]  = Right + Jump + Run
          [10] = Left + Jump + Run
          [11] = Crouch/Down
        
        Args:
            action_id: Action identifier (0-11)
            
        Returns:
            Button state dictionary
        """
        action_map = {
            0:  {},                                          # No action
            1:  {'right': True},                             # Right
            2:  {'left': True},                              # Left
            3:  {'A': True},                                 # Jump
            4:  {'right': True, 'A': True},                  # Right + Jump
            5:  {'left': True, 'A': True},                   # Left + Jump
            6:  {'B': True},                                 # Run/Fire
            7:  {'right': True, 'B': True},                  # Right + Run
            8:  {'left': True, 'B': True},                   # Left + Run
            9:  {'right': True, 'A': True, 'B': True},      # Right + Jump + Run
            10: {'left': True, 'A': True, 'B': True},       # Left + Jump + Run
            11: {'down': True},                              # Crouch/Down
        }
        
        return action_map.get(action_id, {})
    
    async def _handle_episode_event(self, data: Dict[str, Any]):
        """Handle episode event from Lua script.
        
        Terminal events (death, level_complete, time_up) mark the current
        episode as ended so the training loop advances to the next episode.
        
        NOTE: Do NOT call episode_manager.start_episode() here for "started"
        events.  Episodes are created exclusively by _start_episode().
        """
        event = data.get('event')
        episode_id = data.get('episode_id')
        
        self.logger.info(f"Lua episode {episode_id} event: {event}")
        
        # Terminal events from Lua: mark the episode as ended
        # Lua sends the terminal game_state frame just before this event,
        # so the reward calculator and replay buffer already have the death data.
        # This handler ensures the training loop sees the episode as finished.
        if event in ('death', 'level_complete', 'time_up'):
            if (self.episode_manager.current_episode and
                    self.episode_manager.current_episode.status.value == "running"):
                from python.environment.episode_manager import EpisodeStatus
                self.episode_manager.current_episode.termination_reason = event
                if event == 'level_complete':
                    self.episode_manager.current_episode.level_completed = True
                    self.episode_manager.current_episode.status = EpisodeStatus.COMPLETED
                elif event == 'death':
                    self.episode_manager.current_episode.status = EpisodeStatus.FAILED
                else:
                    self.episode_manager.current_episode.status = EpisodeStatus.TERMINATED
                self.logger.info(
                    f"Episode marked terminal by Lua event: {event} "
                    f"(x={self.episode_manager.current_episode.max_x_reached})"
                )
    
    async def _handle_frame_advance(self, data: Dict[str, Any]):
        """Handle frame advance notification."""
        frame_id = data.get('frame_id')
        # Frame synchronization is handled automatically by the WebSocket server
    
    async def _handle_screen_frame(self, frame_id: int, gd_data: bytes):
        """
        Handle screen frame captured by Lua's gui.gdscreenshot().
        
        Decodes GD format image and pushes it into the frame capture buffer.
        This replaces Win32 GDI window capture with direct emulator pixel access.
        
        Args:
            frame_id: Frame identifier from Lua
            gd_data: Raw GD format image bytes
        """
        try:
            self.frame_capture.handle_lua_screen_frame(gd_data)
        except Exception as e:
            self.logger.error(f"Error processing Lua screen frame {frame_id}: {e}")
    
    async def _handle_lua_error(self, data: Dict[str, Any]):
        """Handle error from Lua script."""
        error_code = data.get('error_code')
        message = data.get('message')
        
        self.logger.error(f"Lua script error [{error_code}]: {message}")
        
        # Log debug event
        self.csv_logger.log_debug_event(
            episode=self.current_episode + 1,
            step=self.current_step,
            event_type="error",
            severity="high",
            component="lua",
            message=f"Lua error [{error_code}]: {message}"
        )
    
    def _update_training_phase(self):
        """Update training phase based on episode count."""
        if self.current_episode < self.training_config.warmup_episodes:
            self.training_phase = TrainingPhase.WARMUP
        else:
            self.training_phase = TrainingPhase.TRAINING
        
        # Update curriculum phase if enabled
        if self.training_config.enable_curriculum:
            curriculum_config = self.config.get('training', {}).get('curriculum', {})
            phases = curriculum_config.get('phases', [])
            
            total_episodes = 0
            for phase in phases:
                total_episodes += phase.get('episodes', 0)
                if self.current_episode < total_episodes:
                    phase_name = phase.get('name', 'unknown')
                    
                    # Apply phase-specific settings
                    if phase.get('epsilon_override') is not None:
                        self.agent.epsilon = phase['epsilon_override']
                    
                    self.training_state.curriculum_phase = phase_name
                    break
    
    def _get_quality_threshold(self) -> int:
        """Calculate the current quality gate threshold from qualifying episode history."""
        qg = self._quality_gate_config
        
        if not self._quality_gate_enabled:
            return 0
        
        # Not enough data yet -- use floor
        if len(self._qualifying_distances) < qg.get('min_qualifying', 20):
            return qg.get('floor_x', 200)
        
        # Calculate percentile of qualifying distances
        percentile = qg.get('percentile', 25)
        threshold = int(np.percentile(self._qualifying_distances, percentile))
        
        # Enforce floor
        return max(threshold, qg.get('floor_x', 200))
    
    def _check_quality_gate(self, max_x: int) -> bool:
        """
        Check if an episode qualifies for training.
        
        Returns True if the episode should be committed to the replay buffer
        and used for training. Returns False if it should be discarded.
        """
        if not self._quality_gate_enabled:
            return True
        
        qg = self._quality_gate_config
        
        # Override 1: Warmup episodes always qualify
        if self.current_episode < qg.get('warmup_bypass', 50):
            return True
        
        # Override 2: Not enough qualifying episodes yet (cold start)
        if len(self._qualifying_distances) < qg.get('min_qualifying', 20):
            return True
        
        # Override 3: Reached near-frontier territory
        if self._session_max_x > 0 and max_x >= self._session_max_x * qg.get('frontier_ratio', 0.8):
            return True
        
        # Override 4: Periodic always-qualify (maintains negative learning signal)
        periodic = qg.get('periodic_every', 20)
        if periodic > 0 and (self.current_episode + 1) % periodic == 0:
            return True
        
        # Standard check: must exceed threshold
        threshold = self._get_quality_threshold()
        return max_x >= threshold
    
    def _should_log_error(self, error_key: str) -> bool:
        """
        Rate-limit repeated error messages.  Returns True when the error
        should actually be emitted to logs, False when it should be silently
        counted.
        
        First ``_error_throttle_burst`` occurrences are logged immediately.
        After that, only one log line per ``_error_throttle_interval`` seconds.
        
        Args:
            error_key: Unique key identifying the error category
            
        Returns:
            True if this error should be logged now
        """
        now = time.time()
        count = self._error_counts.get(error_key, 0)
        last_logged = self._error_last_logged.get(error_key, 0.0)
        
        self._error_counts[error_key] = count + 1
        
        # Always log the first few occurrences
        if count < self._error_throttle_burst:
            self._error_last_logged[error_key] = now
            self._error_counts[error_key] = 0  # reset counter after logging
            return True
        
        # After burst, only log every N seconds
        if now - last_logged >= self._error_throttle_interval:
            self._error_last_logged[error_key] = now
            self._error_counts[error_key] = 0  # reset counter after logging
            return True
        
        return False
    
    def _update_fps_tracking(self, step_start_time: float):
        """Update FPS tracking."""
        self.frame_times.append(step_start_time)
        
        # Calculate FPS every second
        if step_start_time - self.last_fps_update >= 1.0:
            if len(self.frame_times) > 1:
                time_span = self.frame_times[-1] - self.frame_times[0]
                if time_span > 0:
                    self.current_fps = len(self.frame_times) / time_span
            
            self.last_fps_update = step_start_time
            self.frame_times.clear()
    
    async def _save_checkpoint(self):
        """Save training checkpoint."""
        try:
            self.logger.info(f"Saving checkpoint at episode {self.current_episode}")
            
            # Get model and optimizer states
            model_state = self.agent.q_network.state_dict()
            optimizer_state = self.agent.optimizer.state_dict()
            
            # Additional data
            additional_data = {
                'current_episode': self.current_episode,
                'current_step': self.current_step,
                'training_phase': self.training_phase.value,
                'session_id': self.session_id
            }
            
            checkpoint_path = self.state_manager.create_checkpoint(
                model_state, optimizer_state, additional_data
            )
            
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    async def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        try:
            self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            
            model_state, optimizer_state, metadata = self.state_manager.load_checkpoint(checkpoint_path)
            
            # Restore model and optimizer
            self.agent.q_network.load_state_dict(model_state)
            self.agent.optimizer.load_state_dict(optimizer_state)
            self.agent.target_network.load_state_dict(model_state)
            
            # Restore training state
            additional_data = metadata.get('additional_data', {})
            self.current_episode = additional_data.get('current_episode', 0)
            self.current_step = additional_data.get('current_step', 0)
            
            self.logger.info(f"Resumed from episode {self.current_episode}")
            
        except Exception as e:
            self.logger.error(f"Failed to resume from checkpoint: {e}")
            raise
    
    async def _run_evaluation(self):
        """Run evaluation episode."""
        self.logger.info("Running evaluation episode...")
        
        # Save current training state
        original_phase = self.training_phase
        original_epsilon = self.agent.epsilon
        
        try:
            # Set evaluation mode
            self.training_phase = TrainingPhase.EVALUATION
            self.agent.epsilon = 0.01  # Minimal exploration
            
            # Skip evaluation for now to prevent disconnection issues
            # The evaluation system needs a complete rewrite to properly handle
            # separate evaluation episodes without interfering with training
            self.logger.info("Evaluation skipped to prevent disconnection issues")
            
        finally:
            # Restore training state
            self.training_phase = original_phase
            self.agent.epsilon = original_epsilon
    
    async def _cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up resources...")
        
        try:
            # Stop frame capture thread
            if self.frame_capture and self.frame_capture.is_capturing:
                self.frame_capture.stop_capture()
                self.logger.info("Frame capture stopped")
            
            # Stop WebSocket server
            if self.websocket_server:
                await self.websocket_server.stop_server()
            
            # Save final checkpoint
            if self.is_running:
                await self._save_checkpoint()
            
            # Close CSV logger
            if self.csv_logger:
                self.csv_logger.close()
            
            # Close TensorBoard writer
            if self.tb_writer is not None:
                self.tb_writer.close()
                self.logger.info("TensorBoard writer closed")
            
            # Export training summary
            if self.state_manager:
                summary = self.state_manager.get_training_summary()
                self.logger.info(f"Training summary: {summary}")
            
            # Create final analysis plot
            if self.plotter:
                analysis_path = self.plotter.create_static_analysis(enable_plotting=self.training_config.enable_plotting)
                if analysis_path:
                    self.logger.info(f"Final analysis saved: {analysis_path}")
            
            self.is_running = False
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def stop_training(self):
        """Stop training gracefully."""
        self.logger.info("Stopping training...")
        self.should_stop = True
        self.is_running = False  # Explicitly set is_running to False to trigger cleanup
        
        # Wait a moment for any pending operations to complete
        await asyncio.sleep(0.1)
        
        self.logger.info("Training stopped")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            'session_id': self.session_id,
            'is_running': self.is_running,
            'training_phase': self.training_phase.value,
            'current_episode': self.current_episode,
            'current_step': self.current_step,
            'current_fps': self.current_fps,
            'websocket_connected': self.websocket_server.is_client_connected() if self.websocket_server else False,
            'agent_stats': self.agent.get_stats() if self.agent else {},
            'system_health': self.health_monitor.get_health_summary() if self.health_monitor else {}
        }


if __name__ == "__main__":
    # Test trainer initialization
    import asyncio
    
    async def test_trainer():
        trainer = MarioTrainer()
        status = trainer.get_training_status()
        print(f"Trainer status: {status}")
        
        # Don't actually start training in test
        print("Trainer initialization test completed!")
    
    asyncio.run(test_trainer())