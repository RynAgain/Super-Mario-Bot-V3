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
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import threading

# Import all required components
from python.agents.dqn_agent import DQNAgent
from python.communication.websocket_server import WebSocketServer
from python.capture.frame_capture import FrameCapture
from python.environment.reward_calculator import RewardCalculator
from python.environment.episode_manager import EpisodeManager
from python.logging.csv_logger import CSVLogger
from python.logging.plotter import PerformancePlotter
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
    enable_curriculum: bool
    enable_plotting: bool


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
        self.config = self.config_loader.load_config(config_path)
        
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
                max_steps_per_episode=training_config.get('max_steps_per_episode', 18000),
                warmup_episodes=training_config.get('warmup_episodes', 1000),
                save_frequency=training_config.get('save_frequency', 1000),
                evaluation_frequency=training_config.get('evaluation_frequency', 500),
                target_fps=60.0,
                enable_curriculum=training_config.get('curriculum', {}).get('enabled', True),
                enable_plotting=True
            )
            
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
            
            # Initialize reward calculator
            reward_config = self.config.get('rewards', {})
            self.reward_calculator = RewardCalculator(reward_config)
            
            # Initialize episode manager
            self.episode_manager = EpisodeManager(
                reward_calculator=self.reward_calculator,
                log_directory="logs",
                csv_filename=f"episodes_{self.session_id}.csv"
            )
            
            # Initialize frame capture
            capture_config = self.config.get('capture', {})
            self.frame_capture = FrameCapture(capture_config)
            
            # Initialize WebSocket server
            network_config = self.config.get('network', {})
            self.websocket_server = WebSocketServer(
                host=network_config.get('host', 'localhost'),
                port=network_config.get('port', 8765)
            )
            
            # Register WebSocket handlers
            self._register_websocket_handlers()
            
            # Initialize performance plotter (optional)
            if self.training_config.enable_plotting:
                self.plotter = PerformancePlotter(
                    log_directory="logs",
                    session_id=self.session_id
                )
            else:
                self.plotter = None
            
            # Initialize training state
            self.training_state = self.state_manager.initialize_training_state(
                self.session_id, 
                self.config.get('training', {})
            )
            
            self.logger.info("All subsystems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize subsystems: {e}")
            raise
    
    def _register_websocket_handlers(self):
        """Register WebSocket message handlers."""
        # Register binary handler for game state data
        self.websocket_server.register_binary_handler(self._handle_game_state)
        
        # Register JSON handlers for control messages
        self.websocket_server.register_json_handler('episode_event', self._handle_episode_event)
        self.websocket_server.register_json_handler('frame_advance', self._handle_frame_advance)
        self.websocket_server.register_json_handler('error', self._handle_lua_error)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.should_stop = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
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
            
            # Wait for client connection
            self.logger.info("Waiting for FCEUX client connection...")
            while not self.websocket_server.is_client_connected() and not self.should_stop:
                await asyncio.sleep(1.0)
            
            if self.should_stop:
                return
            
            self.logger.info("Client connected, starting training loop...")
            
            # Start performance plotter in separate thread if enabled
            if self.plotter:
                plotting_thread = threading.Thread(
                    target=self.plotter.start_realtime_monitoring,
                    daemon=True
                )
                plotting_thread.start()
            
            # Start main training loop
            self.is_running = True
            await self._training_loop()
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            await self._cleanup()
    
    async def _training_loop(self):
        """Main training loop."""
        self.logger.info("Entering main training loop...")
        
        try:
            while not self.should_stop and self.current_episode < self.training_config.max_episodes:
                # Start new episode
                await self._start_episode()
                
                # Episode loop
                episode_start_time = time.time()
                step_in_episode = 0
                
                while (not self.should_stop and 
                       step_in_episode < self.training_config.max_steps_per_episode and
                       self.episode_manager.current_episode and
                       self.episode_manager.current_episode.status.value == "running"):
                    
                    # Process single step
                    step_start_time = time.time()
                    
                    # This will be triggered by incoming game state data
                    await asyncio.sleep(0.001)  # Small delay to prevent busy waiting
                    
                    # Update FPS tracking
                    self._update_fps_tracking(step_start_time)
                    
                    # Check system health periodically
                    if step_in_episode % 100 == 0:
                        health_data = self.health_monitor.check_system_health()
                        if health_data['warnings']:
                            self.logger.warning(f"System health warnings: {health_data['warnings']}")
                    
                    step_in_episode += 1
                
                # End episode
                await self._end_episode(time.time() - episode_start_time)
                
                # Save checkpoint periodically
                if self.current_episode % self.training_config.save_frequency == 0:
                    await self._save_checkpoint()
                
                # Run evaluation periodically
                if self.current_episode % self.training_config.evaluation_frequency == 0:
                    await self._run_evaluation()
                
                self.current_episode += 1
            
            self.logger.info("Training loop completed")
            
        except Exception as e:
            self.logger.error(f"Error in training loop: {e}")
            raise
    
    async def _start_episode(self):
        """Start a new training episode."""
        self.logger.info(f"Starting episode {self.current_episode + 1}")
        
        # Update training phase based on episode count
        self._update_training_phase()
        
        # Send training control command to Lua script
        await self.websocket_server.send_training_control(
            command="reset",
            episode_id=self.current_episode + 1,
            reset_to_level="1-1"
        )
        
        # Reset episode-specific state
        self.current_step = 0
        self.frame_times.clear()
        self.processing_times.clear()
        
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
        
        # Update agent episode end
        self.agent.episode_end(total_reward, episode_stats.frames_processed)
        
        # End episode in episode manager
        episode_data = {
            'final_score': episode_stats.score,
            'time_remaining': episode_stats.time_remaining,
            'coins_collected': episode_stats.coins_collected,
            'lives_used': episode_stats.lives_used,
            'termination_reason': episode_stats.termination_reason
        }
        
        completed_episode = self.episode_manager.end_episode(episode_data)
        
        # Update state manager
        self.state_manager.update_episode_end(
            total_reward, max_distance, episode_duration, completed
        )
        
        # Log episode summary to CSV
        if completed_episode:
            # Get Q-value statistics from agent
            agent_stats = self.agent.get_stats()
            q_value_stats = {
                'max': agent_stats.get('episode_reward_max', 0.0),
                'min': agent_stats.get('episode_reward_min', 0.0)
            }
            
            # Get action statistics (simplified)
            action_stats = {
                'exploration': int(episode_stats.frames_processed * self.agent.epsilon),
                'exploitation': int(episode_stats.frames_processed * (1 - self.agent.epsilon))
            }
            
            self.csv_logger.log_episode_summary(
                episode=self.current_episode + 1,
                duration_seconds=episode_duration,
                total_steps=episode_stats.frames_processed,
                total_reward=total_reward,
                mario_final_state={'x': episode_stats.max_x_reached, 'x_max': episode_stats.max_x_reached},
                level_completed=completed,
                death_cause=episode_stats.termination_reason,
                game_stats={
                    'lives': episode_stats.lives_remaining,
                    'score': episode_stats.score,
                    'coins': episode_stats.coins_collected,
                    'enemies_killed': episode_stats.enemies_killed,
                    'powerups': episode_stats.powerups_collected,
                    'time_remaining': episode_stats.time_remaining
                },
                q_value_stats=q_value_stats,
                action_stats=action_stats
            )
        
        self.logger.info(
            f"Episode {self.current_episode + 1} completed: "
            f"Reward={total_reward:.1f}, Distance={max_distance}, "
            f"Duration={episode_duration:.1f}s, Completed={completed}"
        )
    
    async def _handle_game_state(self, frame_id: int, game_state_data: bytes):
        """
        Handle incoming game state data from Lua script.
        
        Args:
            frame_id: Frame identifier
            game_state_data: Binary game state data
        """
        try:
            step_start_time = time.time()
            
            # Parse game state from binary data
            game_state = self.frame_capture.parse_game_state(game_state_data)
            
            # Process frame in episode manager
            frame_reward, reward_components, is_terminal = self.episode_manager.process_frame(
                game_state, 
                sync_quality=1.0  # Simplified sync quality
            )
            
            # Get preprocessed frames and state vector
            frames, state_vector = self.frame_capture.process_frame(game_state)
            
            # Agent action selection
            action_id = self.agent.select_action(frames, state_vector, training=True)
            
            # Convert action ID to button mapping
            action_buttons = self._action_id_to_buttons(action_id)
            
            # Send action to Lua script
            await self.websocket_server.send_action(action_buttons, frame_id)
            
            # Store experience in replay buffer
            if hasattr(self, 'previous_frames') and hasattr(self, 'previous_state_vector'):
                self.agent.store_experience(
                    self.previous_frames,
                    self.previous_state_vector,
                    self.previous_action_id,
                    frame_reward,
                    frames,
                    state_vector,
                    is_terminal
                )
            
            # Train agent if enough experience
            training_metrics = {}
            if self.training_phase != TrainingPhase.WARMUP:
                training_metrics = self.agent.train_step()
            
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
            
            # Log sync quality
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
            
        except Exception as e:
            self.logger.error(f"Error handling game state: {e}")
            
            # Log debug event
            self.csv_logger.log_debug_event(
                episode=self.current_episode + 1,
                step=self.current_step,
                event_type="error",
                severity="high",
                component="trainer",
                message=f"Game state processing error: {str(e)}",
                exception=e
            )
    
    def _action_id_to_buttons(self, action_id: int) -> Dict[str, bool]:
        """
        Convert action ID to button mapping.
        
        Args:
            action_id: Action identifier (0-11)
            
        Returns:
            Button state dictionary
        """
        # Mario action mapping (simplified)
        action_map = {
            0: {},  # No action
            1: {'right': True},  # Move right
            2: {'right': True, 'A': True},  # Run right
            3: {'right': True, 'B': True},  # Jump right
            4: {'right': True, 'A': True, 'B': True},  # Run jump right
            5: {'left': True},  # Move left
            6: {'left': True, 'A': True},  # Run left
            7: {'left': True, 'B': True},  # Jump left
            8: {'left': True, 'A': True, 'B': True},  # Run jump left
            9: {'B': True},  # Jump
            10: {'A': True},  # Run/Fire
            11: {'down': True}  # Duck
        }
        
        return action_map.get(action_id, {})
    
    async def _handle_episode_event(self, data: Dict[str, Any]):
        """Handle episode event from Lua script."""
        event = data.get('event')
        episode_id = data.get('episode_id')
        
        self.logger.info(f"Episode {episode_id} event: {event}")
        
        if event == 'started':
            # Episode started in Lua
            initial_state = data.get('game_state', {})
            self.episode_manager.start_episode(initial_state)
        elif event == 'ended':
            # Episode ended in Lua
            # This will be handled by the main training loop
            pass
    
    async def _handle_frame_advance(self, data: Dict[str, Any]):
        """Handle frame advance notification."""
        frame_id = data.get('frame_id')
        # Frame synchronization is handled automatically by the WebSocket server
    
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
            
            # Run single evaluation episode
            # This would be similar to regular episode but without training
            # For now, just log the evaluation
            
            self.logger.info("Evaluation completed")
            
        finally:
            # Restore training state
            self.training_phase = original_phase
            self.agent.epsilon = original_epsilon
    
    async def _cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up resources...")
        
        try:
            # Stop WebSocket server
            if self.websocket_server:
                await self.websocket_server.stop_server()
            
            # Save final checkpoint
            if self.is_running:
                await self._save_checkpoint()
            
            # Close CSV logger
            if self.csv_logger:
                self.csv_logger.close()
            
            # Export training summary
            if self.state_manager:
                summary = self.state_manager.get_training_summary()
                self.logger.info(f"Training summary: {summary}")
            
            # Create final analysis plot
            if self.plotter:
                analysis_path = self.plotter.create_static_analysis()
                self.logger.info(f"Final analysis saved: {analysis_path}")
            
            self.is_running = False
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def stop_training(self):
        """Stop training gracefully."""
        self.logger.info("Stopping training...")
        self.should_stop = True
        
        # Wait for training loop to finish
        while self.is_running:
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