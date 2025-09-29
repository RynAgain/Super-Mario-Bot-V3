"""
DQN Agent for Super Mario Bros AI Training

This module implements the DQN agent with:
- Experience replay buffer integration
- Target network with soft updates
- Epsilon-greedy exploration with decay
- Double DQN implementation
- Huber loss function
- Gradient clipping
- Mixed precision training support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from collections import deque
import copy

from python.models.dueling_dqn import DuelingDQN, create_dueling_dqn
from python.utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from python.utils.preprocessing import MarioPreprocessor
from python.utils.model_utils import DeviceManager, ModelOptimizer, ModelManager


class DQNAgent:
    """
    Deep Q-Network Agent for Super Mario Bros.
    
    Implements Double DQN with Dueling architecture, experience replay,
    target networks, and various training optimizations.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "auto"
    ):
        """
        Initialize DQN Agent.
        
        Args:
            config: Configuration dictionary containing all hyperparameters
            device: Device for training ("auto", "cpu", "cuda")
        """
        self.config = config
        
        # Setup device management
        self.device_manager = DeviceManager(device)
        self.device = self.device_manager.device
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 0.00025)
        self.batch_size = config.get('batch_size', 32)
        self.gamma = config.get('gamma', 0.99)
        self.target_update_frequency = config.get('target_update_frequency', 1000)
        self.gradient_clipping = config.get('gradient_clipping', 10.0)
        
        # Exploration parameters
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_decay_type = config.get('epsilon_decay_type', 'exponential')
        
        # Training options
        self.double_dqn = config.get('double_dqn', True)
        self.prioritized_replay = config.get('prioritized_replay', False)
        self.mixed_precision = config.get('mixed_precision', True)
        
        # Initialize networks
        self._initialize_networks()
        
        # Initialize replay buffer
        self._initialize_replay_buffer()
        
        # Initialize optimizer
        self._initialize_optimizer()
        
        # Initialize preprocessor
        self.preprocessor = MarioPreprocessor(device=str(self.device))
        
        # Training state
        self.episode = 0
        self.step = 0
        self.training_step = 0
        self.last_target_update = 0
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_losses = deque(maxlen=100)
        self.episode_q_values = deque(maxlen=100)
        
        # Model management
        self.model_manager = ModelManager(
            checkpoint_dir=config.get('checkpoint_dir', 'checkpoints'),
            max_checkpoints=config.get('max_checkpoints', 5)
        )
        
        self.logger.info("DQN Agent initialized successfully")
    
    def _initialize_networks(self):
        """Initialize main and target networks."""
        # Create main network
        self.q_network = create_dueling_dqn()
        
        # Create target network (copy of main network)
        self.target_network = create_dueling_dqn()
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Move networks to device and optimize
        optimizer = ModelOptimizer(self.device_manager)
        self.q_network = optimizer.optimize_model(
            self.q_network,
            compile_model=self.config.get('compile_model', True),
            mixed_precision=self.mixed_precision
        )
        self.target_network = self.device_manager.to_device(self.target_network)
        
        # Set target network to eval mode
        self.target_network.eval()
        
        # Create gradient scaler for mixed precision
        self.scaler = optimizer.create_scaler() if self.mixed_precision else None
        
        self.logger.info(f"Networks initialized on device: {self.device}")
    
    def _initialize_replay_buffer(self):
        """Initialize experience replay buffer."""
        buffer_config = {
            'capacity': self.config.get('replay_buffer_size', 20000),  # more stable gradients
            'device': str(self.device),
            'frame_stack_size': 4,  # Fixed: changed from 8 to 4 to match model
            'frame_size': (84, 84),
            'state_vector_size': 12
        }
        
        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                **buffer_config,
                alpha=self.config.get('priority_alpha', 0.6),
                beta=self.config.get('priority_beta', 0.4),
                beta_increment=self.config.get('priority_beta_increment', 0.001)
            )
            self.logger.info("Using prioritized experience replay")
        else:
            self.replay_buffer = ReplayBuffer(**buffer_config)
            self.logger.info("Using uniform experience replay")
    
    def _initialize_optimizer(self):
        """Initialize optimizer."""
        optimizer_type = self.config.get('optimizer', 'Adam')
        
        if optimizer_type == 'Adam':
            self.optimizer = optim.Adam(
                self.q_network.parameters(),
                lr=self.learning_rate,
                weight_decay=self.config.get('weight_decay', 0.0001)
            )
        elif optimizer_type == 'RMSprop':
            self.optimizer = optim.RMSprop(
                self.q_network.parameters(),
                lr=self.learning_rate,
                weight_decay=self.config.get('weight_decay', 0.0001)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        self.logger.info(f"Optimizer initialized: {optimizer_type}")
    
    def select_action(
        self,
        frames: torch.Tensor,
        state_vector: torch.Tensor,
        training: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            frames: Stacked frames tensor (1, 4, 84, 84)
            state_vector: Game state vector (1, 12)
            training: Whether in training mode
            
        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            # Random action (exploration)
            return np.random.randint(0, 12)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                q_values = self.q_network(frames, state_vector)
                return q_values.argmax(dim=1).item()
    
    def store_experience(
        self,
        state_frames: torch.Tensor,
        state_vector: torch.Tensor,
        action: int,
        reward: float,
        next_state_frames: torch.Tensor,
        next_state_vector: torch.Tensor,
        done: bool
    ):
        """
        Store experience in replay buffer.
        
        Args:
            state_frames: Current frame stack
            state_vector: Current state vector
            action: Action taken
            reward: Reward received
            next_state_frames: Next frame stack
            next_state_vector: Next state vector
            done: Episode termination flag
        """
        self.replay_buffer.add(
            state_frames.squeeze(0).cpu(),  # Remove batch dimension
            state_vector.squeeze(0).cpu(),
            action,
            reward,
            next_state_frames.squeeze(0).cpu(),
            next_state_vector.squeeze(0).cpu(),
            done
        )
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.
        
        Returns:
            Dictionary containing training metrics
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return {}
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        (state_frames, state_vectors, actions, rewards,
         next_state_frames, next_state_vectors, dones, weights, indices) = batch
        
        # Compute loss
        if self.mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                loss, td_errors = self._compute_loss(
                    state_frames, state_vectors, actions, rewards,
                    next_state_frames, next_state_vectors, dones, weights
                )
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.gradient_clipping > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.q_network.parameters(),
                    self.gradient_clipping
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, td_errors = self._compute_loss(
                state_frames, state_vectors, actions, rewards,
                next_state_frames, next_state_vectors, dones, weights
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.q_network.parameters(),
                    self.gradient_clipping
                )
            
            self.optimizer.step()
        
        # Update priorities for prioritized replay
        if self.prioritized_replay:
            priorities = torch.abs(td_errors) + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)
        
        # Update target network if needed
        if self.training_step - self.last_target_update >= self.target_update_frequency:
            self._update_target_network()
            self.last_target_update = self.training_step
        
        # Update exploration rate
        self._update_epsilon()
        
        self.training_step += 1
        
        # Return metrics
        with torch.no_grad():
            current_q_values = self.q_network(state_frames, state_vectors)
            mean_q_value = current_q_values.mean().item()
        
        return {
            'loss': loss.item(),
            'mean_q_value': mean_q_value,
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }
    
    def _compute_loss(
        self,
        state_frames: torch.Tensor,
        state_vectors: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_state_frames: torch.Tensor,
        next_state_vectors: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute DQN loss with optional Double DQN.
        
        Returns:
            Tuple of (loss, td_errors)
        """
        # Current Q-values
        current_q_values = self.q_network(state_frames, state_vectors)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_q_values_main = self.q_network(next_state_frames, next_state_vectors)
                next_actions = next_q_values_main.argmax(dim=1)
                
                next_q_values_target = self.target_network(next_state_frames, next_state_vectors)
                next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_state_frames, next_state_vectors)
                next_q_values = next_q_values.max(dim=1)[0]
            
            # Compute target Q-values
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        
        # Compute TD errors
        td_errors = target_q_values - current_q_values
        
        # Compute loss (Huber loss)
        loss_function = self.config.get('loss_function', 'Huber')
        if loss_function == 'Huber':
            loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
        elif loss_function == 'MSE':
            loss = F.mse_loss(current_q_values, target_q_values, reduction='none')
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        # Apply importance sampling weights
        weighted_loss = (loss * weights).mean()
        
        return weighted_loss, td_errors.detach()
    
    def _update_target_network(self):
        """Update target network with current network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.logger.debug(f"Target network updated at step {self.training_step}")
    
    def _update_epsilon(self):
        """Update exploration rate."""
        if self.epsilon_decay_type == 'exponential':
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon * self.epsilon_decay
            )
        elif self.epsilon_decay_type == 'linear':
            decay_steps = self.config.get('epsilon_decay_steps', 50000)
            decay_amount = (self.epsilon_start - self.epsilon_end) / decay_steps
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon - decay_amount
            )
    
    def episode_end(self, total_reward: float, episode_length: int):
        """
        Called at the end of each episode.
        
        Args:
            total_reward: Total reward for the episode
            episode_length: Number of steps in the episode
        """
        self.episode += 1
        self.episode_rewards.append(total_reward)
        
        # Reset preprocessor for new episode
        self.preprocessor.reset()
        
        self.logger.info(
            f"Episode {self.episode} completed: "
            f"Reward={total_reward:.1f}, Length={episode_length}, "
            f"Epsilon={self.epsilon:.3f}"
        )
    
    def save_checkpoint(self, metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Save agent checkpoint.
        
        Args:
            metrics: Optional metrics to save with checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_metrics = {
            'episode_reward_mean': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'episode_reward_std': np.std(self.episode_rewards) if self.episode_rewards else 0.0,
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }
        
        if metrics:
            checkpoint_metrics.update(metrics)
        
        return self.model_manager.save_checkpoint(
            model=self.q_network,
            optimizer=self.optimizer,
            episode=self.episode,
            step=self.step,
            metrics=checkpoint_metrics,
            metadata={
                'config': self.config,
                'replay_buffer_size': len(self.replay_buffer)
            }
        )
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load agent checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint metadata
        """
        metadata = self.model_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.q_network,
            optimizer=self.optimizer,
            device=str(self.device)
        )
        
        # Restore training state
        self.episode = metadata.get('episode', 0)
        self.step = metadata.get('step', 0)
        self.training_step = metadata.get('training_step', 0)
        
        # Update target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.logger.info(f"Loaded checkpoint from episode {self.episode}")
        
        return metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current training statistics.
        
        Returns:
            Dictionary of training statistics
        """
        stats = {
            'episode': self.episode,
            'step': self.step,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'replay_buffer_size': len(self.replay_buffer),
            'replay_buffer_capacity': self.replay_buffer.capacity
        }
        
        if self.episode_rewards:
            stats.update({
                'episode_reward_mean': np.mean(self.episode_rewards),
                'episode_reward_std': np.std(self.episode_rewards),
                'episode_reward_min': np.min(self.episode_rewards),
                'episode_reward_max': np.max(self.episode_rewards)
            })
        
        if self.episode_losses:
            stats.update({
                'loss_mean': np.mean(self.episode_losses),
                'loss_std': np.std(self.episode_losses)
            })
        
        # Memory usage
        memory_info = self.device_manager.get_memory_info()
        stats.update(memory_info)
        
        return stats


if __name__ == "__main__":
    # Test DQN agent
    config = {
        'learning_rate': 0.00025,
        'batch_size': 32,
        'gamma': 0.99,
        'target_update_frequency': 1000,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'replay_buffer_size': 10000,
        'double_dqn': True,
        'mixed_precision': True,
        'compile_model': False  # Disable for testing
    }
    
    print("Testing DQN Agent...")
    
    # Create agent
    agent = DQNAgent(config, device="cpu")  # Use CPU for testing
    
    # Test action selection
    dummy_frames = torch.randn(1, 4, 84, 84)  # Fixed: changed from 8 to 4
    dummy_state = torch.randn(1, 12)
    
    action = agent.select_action(dummy_frames, dummy_state, training=True)
    print(f"Selected action: {action}")
    
    # Test experience storage
    agent.store_experience(
        dummy_frames, dummy_state, action, 1.0,
        dummy_frames, dummy_state, False
    )
    
    # Add more experiences
    for _ in range(100):
        frames = torch.randn(1, 4, 84, 84)  # Fixed: changed from 8 to 4
        state = torch.randn(1, 12)
        action = np.random.randint(0, 12)
        reward = np.random.randn()
        done = np.random.random() < 0.1
        
        agent.store_experience(frames, state, action, reward, frames, state, done)
    
    # Test training step
    if agent.replay_buffer.is_ready(agent.batch_size):
        metrics = agent.train_step()
        print(f"Training metrics: {metrics}")
    
    # Test statistics
    stats = agent.get_stats()
    print(f"Agent stats: {stats}")
    
    print("DQN Agent tests completed!")