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
        
        # Gradient accumulation -- effective batch = batch_size * accumulation_steps
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self._grad_accum_counter = 0
        
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
        
        # N-step returns buffer
        self.n_step = config.get('n_step', 3)
        self.n_step_buffer = deque(maxlen=self.n_step)
        
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
        from python.models.dueling_dqn import DuelingDQNConfig
        
        # Build config from training config
        model_config = DuelingDQNConfig()
        model_config.noisy = self.config.get('noisy_networks', True)
        model_config.num_atoms = self.config.get('num_atoms', 51)
        model_config.v_min = self.config.get('v_min', -10.0)
        model_config.v_max = self.config.get('v_max', 10.0)
        
        # Store C51 flag for loss dispatch
        self.distributional = model_config.num_atoms > 1
        self.num_atoms = model_config.num_atoms
        self.v_min = model_config.v_min
        self.v_max = model_config.v_max
        
        # Create main network
        self.q_network = create_dueling_dqn(model_config)
        
        # Create target network (copy of main network)
        self.target_network = create_dueling_dqn(model_config)
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
        Select action using epsilon-greedy OR NoisyNet exploration.
        
        With NoisyNet enabled, epsilon-greedy is bypassed -- the noise in
        the network weights provides state-dependent exploration automatically.
        
        Args:
            frames: Stacked frames tensor (1, 4, 84, 84)
            state_vector: Game state vector (1, 12)
            training: Whether in training mode
            
        Returns:
            Selected action index
        """
        # NoisyNet: exploration comes from network noise, not epsilon
        use_noisy = getattr(self.q_network, 'noisy', False)
        
        if not use_noisy and training and np.random.random() < self.epsilon:
            # Epsilon-greedy random action (only when NoisyNet is disabled)
            return np.random.randint(0, 12)
        else:
            with torch.no_grad():
                # Reset noise before each forward pass for fresh exploration
                if use_noisy and training:
                    self.q_network.reset_noise()
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
        Store experience using n-step returns.
        
        Buffers N transitions and computes discounted n-step return:
          R_n = r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^(n-1) * r_{n-1}
        
        The replay buffer stores (s_0, a_0, R_n, s_n, done_n) instead of
        single-step transitions, enabling faster value propagation.
        
        Args:
            state_frames: Current frame stack
            state_vector: Current state vector
            action: Action taken
            reward: Reward received
            next_state_frames: Next frame stack
            next_state_vector: Next state vector
            done: Episode termination flag
        """
        # Add transition to n-step buffer
        self.n_step_buffer.append((
            state_frames.squeeze(0).cpu(),
            state_vector.squeeze(0).cpu(),
            action,
            reward,
            next_state_frames.squeeze(0).cpu(),
            next_state_vector.squeeze(0).cpu(),
            done
        ))
        
        # Flush buffer when full or on terminal state
        if len(self.n_step_buffer) == self.n_step or done:
            # Compute n-step discounted return
            n_step_return = 0.0
            for i in reversed(range(len(self.n_step_buffer))):
                _, _, _, r, _, _, d = self.n_step_buffer[i]
                n_step_return = r + self.gamma * n_step_return * (not d)
            
            # Use first transition's state and action, last transition's next_state
            first = self.n_step_buffer[0]
            last = self.n_step_buffer[-1]
            
            self.replay_buffer.add(
                first[0],  # state_frames from step 0
                first[1],  # state_vector from step 0
                first[2],  # action from step 0
                n_step_return,  # n-step discounted return
                last[4],   # next_state_frames from step n
                last[5],   # next_state_vector from step n
                last[6]    # done flag from step n
            )
            
            # On terminal state, flush all remaining partial sequences
            if done:
                # Store remaining partial n-step returns
                for start_idx in range(1, len(self.n_step_buffer)):
                    partial_return = 0.0
                    for i in reversed(range(start_idx, len(self.n_step_buffer))):
                        _, _, _, r, _, _, d = self.n_step_buffer[i]
                        partial_return = r + self.gamma * partial_return * (not d)
                    
                    entry = self.n_step_buffer[start_idx]
                    self.replay_buffer.add(
                        entry[0], entry[1], entry[2],
                        partial_return,
                        last[4], last[5], last[6]
                    )
                
                self.n_step_buffer.clear()
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step with optional gradient accumulation.
        
        When ``gradient_accumulation_steps`` > 1, gradients are accumulated
        across multiple mini-batches before the optimizer steps.  This gives
        an effective batch size of ``batch_size * gradient_accumulation_steps``
        without increasing GPU memory usage.
        
        Returns:
            Dictionary containing training metrics (empty dict on accumulation
            sub-steps that don't trigger an optimizer update)
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return {}
        
        # Reset noise for training forward pass (NoisyNet)
        if getattr(self.q_network, 'noisy', False):
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        (state_frames, state_vectors, actions, rewards,
         next_state_frames, next_state_vectors, dones, weights, indices) = batch
        
        accum = self.gradient_accumulation_steps
        
        # --- Gradient accumulation: zero grads only on first sub-step ---
        if self._grad_accum_counter == 0:
            self.optimizer.zero_grad()
        
        # Compute loss (scaled by 1/accum so accumulated gradient is correct)
        if self.mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                loss, td_errors = self._compute_loss(
                    state_frames, state_vectors, actions, rewards,
                    next_state_frames, next_state_vectors, dones, weights
                )
            scaled_loss = loss / accum
            self.scaler.scale(scaled_loss).backward()
        else:
            loss, td_errors = self._compute_loss(
                state_frames, state_vectors, actions, rewards,
                next_state_frames, next_state_vectors, dones, weights
            )
            scaled_loss = loss / accum
            scaled_loss.backward()
        
        self._grad_accum_counter += 1
        
        # --- Only step optimizer after accumulating enough sub-steps ---
        if self._grad_accum_counter < accum:
            # Update priorities even on sub-steps so PER stays current
            if self.prioritized_replay:
                priorities = torch.abs(td_errors) + 1e-6
                self.replay_buffer.update_priorities(indices, priorities)
            return {}  # no metrics until optimizer steps
        
        # Reset counter for next accumulation window
        self._grad_accum_counter = 0
        
        # Gradient clipping + optimizer step
        if self.mixed_precision and self.scaler is not None:
            if self.gradient_clipping > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.q_network.parameters(),
                    self.gradient_clipping
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
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
        
        # Update target network (soft update every step, hard update periodically)
        tau = self.config.get('tau', 0.0)
        if tau > 0:
            # Soft update (Polyak averaging) every step
            self._soft_update_target_network(tau)
        elif self.training_step - self.last_target_update >= self.target_update_frequency:
            # Hard update periodically
            self._update_target_network()
            self.last_target_update = self.training_step
        
        # Do NOT update epsilon here -- it is now updated per-episode in episode_end()
        
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
        Compute DQN loss -- dispatches to distributional (C51) or standard path.
        
        Returns:
            Tuple of (loss, td_errors)
        """
        if self.distributional:
            return self._compute_distributional_loss(
                state_frames, state_vectors, actions, rewards,
                next_state_frames, next_state_vectors, dones, weights
            )
        
        return self._compute_standard_loss(
            state_frames, state_vectors, actions, rewards,
            next_state_frames, next_state_vectors, dones, weights
        )
    
    def _compute_standard_loss(
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
        Standard (non-distributional) DQN loss with optional Double DQN.
        
        Returns:
            Tuple of (loss, td_errors)
        """
        # Current Q-values
        current_q_values = self.q_network(state_frames, state_vectors)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values
        with torch.no_grad():
            if self.double_dqn:
                next_q_values_main = self.q_network(next_state_frames, next_state_vectors)
                next_actions = next_q_values_main.argmax(dim=1)
                next_q_values_target = self.target_network(next_state_frames, next_state_vectors)
                next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_network(next_state_frames, next_state_vectors)
                next_q_values = next_q_values.max(dim=1)[0]
            
            gamma_n = self.gamma ** self.n_step
            target_q_values = rewards + (gamma_n * next_q_values * (~dones))
        
        td_errors = target_q_values - current_q_values
        
        loss_function = self.config.get('loss_function', 'Huber')
        if loss_function == 'Huber':
            loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
        elif loss_function == 'MSE':
            loss = F.mse_loss(current_q_values, target_q_values, reduction='none')
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        weighted_loss = (loss * weights).mean()
        return weighted_loss, td_errors.detach()
    
    def _compute_distributional_loss(
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
        C51 distributional cross-entropy loss.
        
        Projects the target distribution onto the fixed support using the
        Bellman operator, then computes KL divergence (cross-entropy) between
        the projected target and the current distribution.
        
        Returns:
            Tuple of (loss, td_errors_for_PER)
        """
        B = state_frames.size(0)
        N = self.num_atoms
        gamma_n = self.gamma ** self.n_step
        
        support = self.q_network.support  # (N,)
        delta_z = self.q_network.delta_z
        
        # Current log-probabilities for chosen actions: (B, N)
        log_probs = self.q_network.forward_dist(state_frames, state_vectors)  # (B, A, N)
        log_probs_a = log_probs[torch.arange(B), actions]  # (B, N)
        
        with torch.no_grad():
            # --- Double DQN action selection ---
            if self.double_dqn:
                # Use main network Q-values (expected) to select best next action
                next_q = self.q_network(next_state_frames, next_state_vectors)  # (B, A)
                next_actions = next_q.argmax(dim=1)  # (B,)
            else:
                next_q = self.target_network(next_state_frames, next_state_vectors)
                next_actions = next_q.argmax(dim=1)
            
            # Target distribution for the selected next action: (B, N)
            target_log_probs = self.target_network.forward_dist(
                next_state_frames, next_state_vectors
            )  # (B, A, N)
            target_probs = target_log_probs[torch.arange(B), next_actions].exp()  # (B, N)
            
            # --- Bellman projection ---
            # T_z = r + gamma^n * z  (clipped to [v_min, v_max])
            Tz = rewards.unsqueeze(1) + gamma_n * (~dones).unsqueeze(1).float() * support.unsqueeze(0)
            Tz = Tz.clamp(self.v_min, self.v_max)  # (B, N)
            
            # Map Tz onto atom indices
            b_idx = (Tz - self.v_min) / delta_z  # (B, N), float in [0, N-1]
            lower = b_idx.floor().long()
            upper = b_idx.ceil().long()
            
            # Clamp to valid range
            lower = lower.clamp(0, N - 1)
            upper = upper.clamp(0, N - 1)
            
            # Distribute probability mass to neighbors
            projected = torch.zeros(B, N, device=state_frames.device)
            
            # Fraction allocated to upper neighbor
            upper_frac = b_idx - lower.float()
            lower_frac = 1.0 - upper_frac
            
            # Scatter-add probabilities
            projected.scatter_add_(1, lower, target_probs * lower_frac)
            projected.scatter_add_(1, upper, target_probs * upper_frac)
        
        # Cross-entropy loss: -sum(m_i * log(p_i))
        loss_per_sample = -(projected * log_probs_a).sum(dim=1)  # (B,)
        
        # TD errors for PER (use expected Q difference as proxy)
        with torch.no_grad():
            current_q = (log_probs_a.exp() * support.unsqueeze(0)).sum(dim=1)
            target_q = (projected * support.unsqueeze(0)).sum(dim=1)
            td_errors = target_q - current_q
        
        weighted_loss = (loss_per_sample * weights).mean()
        return weighted_loss, td_errors.detach()
    
    def _update_target_network(self):
        """Hard update: copy main network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.logger.debug(f"Target network hard-updated at step {self.training_step}")
    
    def _soft_update_target_network(self, tau: float):
        """
        Soft update: blend main network weights into target network.
        
        target = tau * main + (1 - tau) * target
        
        Args:
            tau: Interpolation factor (0.005 is typical)
        """
        for target_param, main_param in zip(self.target_network.parameters(),
                                            self.q_network.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)
    
    def _update_epsilon(self):
        """
        Update exploration rate. Called per-EPISODE (not per step).
        
        Per-step decay with rate 0.9995 takes ~6000 steps to reach 0.05,
        which means epsilon barely moves during early training.
        Per-episode decay with rate 0.998 reaches 0.05 in ~1500 episodes.
        """
        if self.epsilon_decay_type == 'exponential':
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon * self.epsilon_decay
            )
        elif self.epsilon_decay_type == 'linear':
            decay_episodes = self.config.get('epsilon_decay_episodes', 3000)
            decay_amount = (self.epsilon_start - self.epsilon_end) / decay_episodes
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
        
        # Clear n-step buffer for new episode
        self.n_step_buffer.clear()
        
        # Decay epsilon ONCE per episode (not per training step)
        self._update_epsilon()
        
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