"""
Dueling DQN Neural Network Model for Super Mario Bros AI

This module implements the Dueling DQN architecture with 4-frame stacking
for temporal understanding and improved action-value estimation.

Architecture:
- Convolutional layers for visual processing (84x84 grayscale frames)
- Frame stacking support (4 frames, optimized for performance)
- Separate value and advantage streams (dueling architecture)
- 12-action output space matching the Lua script
- GPU acceleration support
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class NoisyLinear(nn.Module):
    """
    Factorized Noisy Linear layer (Fortunato et al., 2018).
    
    Replaces nn.Linear with learnable noise parameters. The noise provides
    state-dependent exploration that self-anneals as the network becomes
    confident about action values.
    
    y = (mu_w + sigma_w * eps_w) @ x + (mu_b + sigma_b * eps_b)
    
    Uses factorized Gaussian noise for efficiency:
      eps_w = f(eps_i) * f(eps_j)^T  where f(x) = sign(x) * sqrt(|x|)
    """
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))
        
        # Factorized noise buffers (not parameters, regenerated each forward pass)
        self.register_buffer('eps_i', torch.zeros(1, in_features))
        self.register_buffer('eps_j', torch.zeros(out_features, 1))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize mu and sigma parameters."""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-mu_range, mu_range)
        self.mu_bias.data.uniform_(-mu_range, mu_range)
        self.sigma_weight.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma_init / math.sqrt(self.out_features))
    
    @staticmethod
    def _f_noise(x: torch.Tensor) -> torch.Tensor:
        """Factorized noise function: f(x) = sign(x) * sqrt(|x|)"""
        return x.sign() * x.abs().sqrt()
    
    def reset_noise(self):
        """Sample new factorized noise."""
        eps_i = self._f_noise(torch.randn(1, self.in_features, device=self.mu_weight.device))
        eps_j = self._f_noise(torch.randn(self.out_features, 1, device=self.mu_weight.device))
        self.eps_i.copy_(eps_i)
        self.eps_j.copy_(eps_j)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Noisy forward: use mu + sigma * noise
            weight_noise = self.eps_j * self.eps_i  # outer product (out, in)
            bias_noise = self.eps_j.squeeze(1)       # (out,)
            weight = self.mu_weight + self.sigma_weight * weight_noise
            bias = self.mu_bias + self.sigma_bias * bias_noise
        else:
            # Deterministic forward: use mu only (no noise during evaluation)
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    """
    Dueling DQN implementation with 4-frame stacking, game state fusion,
    and optional C51 distributional output.
    
    The network processes stacked frames through convolutional layers,
    fuses them with game state features, and uses dueling architecture
    to separate value and advantage estimation.
    
    When ``num_atoms > 1`` (C51 mode), the value and advantage streams
    output probability distributions over a discrete set of return atoms
    instead of scalar values. Q-values are recovered as the expected
    value: Q(s,a) = sum_i(z_i * p_i(s,a)).
    """
    
    def __init__(
        self,
        num_actions: int = 12,
        state_vector_size: int = 12,
        frame_stack_size: int = 4,
        frame_size: Tuple[int, int] = (84, 84),
        noisy: bool = False,
        sigma_init: float = 0.5,
        # C51 distributional parameters
        num_atoms: int = 1,
        v_min: float = -10.0,
        v_max: float = 10.0
    ):
        """
        Initialize the Dueling DQN model.
        
        Args:
            num_actions: Number of possible actions (12 for Mario)
            state_vector_size: Size of game state vector (12 features)
            frame_stack_size: Number of frames to stack (4 frames)
            frame_size: Frame dimensions (height, width) = (84, 84)
            noisy: Use NoisyLinear for exploration (replaces epsilon-greedy)
            sigma_init: Initial noise magnitude for NoisyLinear
            num_atoms: Number of atoms for C51 distributional output (1 = standard DQN)
            v_min: Minimum return value for C51 support
            v_max: Maximum return value for C51 support
        """
        super(DuelingDQN, self).__init__()
        
        self.num_actions = num_actions
        self.state_vector_size = state_vector_size
        self.frame_stack_size = frame_stack_size
        self.frame_size = frame_size
        self.noisy = noisy
        
        # C51 distributional parameters
        self.num_atoms = num_atoms
        self.distributional = num_atoms > 1
        self.v_min = v_min
        self.v_max = v_max
        if self.distributional:
            self.register_buffer(
                'support', torch.linspace(v_min, v_max, num_atoms)
            )
            self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Select linear layer type
        LinearLayer = NoisyLinear if noisy else nn.Linear
        linear_kwargs = {'sigma_init': sigma_init} if noisy else {}
        
        # Convolutional layers for frame processing (always standard -- no noise needed here)
        self.conv1 = nn.Conv2d(
            in_channels=frame_stack_size,
            out_channels=32,
            kernel_size=8,
            stride=4,
            padding=2
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1
        )
        
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Calculate convolutional output size
        self.conv_output_size = self._calculate_conv_output_size()
        
        # Feature fusion layer (noisy if enabled)
        self.fusion_fc = LinearLayer(
            self.conv_output_size + state_vector_size,
            512,
            **linear_kwargs
        )
        
        # Value stream outputs num_atoms when distributional, 1 otherwise
        value_out = num_atoms if self.distributional else 1
        self.value_fc1 = LinearLayer(512, 256, **linear_kwargs)
        self.value_fc2 = LinearLayer(256, value_out, **linear_kwargs)
        
        # Advantage stream outputs num_actions * num_atoms when distributional
        advantage_out = num_actions * num_atoms if self.distributional else num_actions
        self.advantage_fc1 = LinearLayer(512, 256, **linear_kwargs)
        self.advantage_fc2 = LinearLayer(256, advantage_out, **linear_kwargs)
        
        # Dropout for regularization (reduced with noisy nets since noise acts as regularizer)
        self.dropout = nn.Dropout(0.1 if noisy else 0.3)
        
        # Initialize weights
        self._initialize_weights()
        
    def _calculate_conv_output_size(self) -> int:
        """
        Calculate the output size after convolutional layers.
        
        Input: (4, 84, 84)
        After conv1 (8x8, stride=4, pad=2): (32, 21, 21)
        After conv2 (4x4, stride=2, pad=1): (64, 11, 11)
        After conv3 (3x3, stride=1, pad=1): (64, 11, 11)
        
        Returns:
            Total flattened size: 64 * 11 * 11 = 7744
        """
        # Simulate forward pass to calculate size
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.frame_stack_size, *self.frame_size)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x.numel() // x.size(0)  # Total elements per batch item
    
    def _initialize_weights(self):
        """Initialize network weights using Kaiming normal initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        frames: torch.Tensor,
        state_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the Dueling DQN.
        
        Args:
            frames: Stacked frames tensor of shape (batch_size, 4, 84, 84)
            state_vector: Game state vector of shape (batch_size, 12)
            
        Returns:
            - Standard mode: Q-values tensor of shape (batch_size, num_actions)
            - C51 mode: Q-values tensor of shape (batch_size, num_actions)
              (expected values derived from distributions).
              Use ``forward_dist()`` to get raw log-probabilities.
        """
        if self.distributional:
            # C51: get distributions, compute expected Q-values
            log_probs = self.forward_dist(frames, state_vector)  # (B, A, N)
            probs = log_probs.exp()
            q_values = (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)
            return q_values
        
        batch_size = frames.size(0)
        
        # Process frame stack through convolutional layers
        x = F.relu(self.conv1(frames))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten convolutional output (ensure contiguous memory layout)
        conv_features = x.contiguous().view(batch_size, -1)
        
        # Fuse convolutional features with state vector
        fused_features = torch.cat([conv_features, state_vector], dim=1)
        fused_features = F.relu(self.fusion_fc(fused_features))
        fused_features = self.dropout(fused_features)
        
        # Value stream - estimates V(s)
        value = F.relu(self.value_fc1(fused_features))
        value = self.dropout(value)
        value = self.value_fc2(value)  # Shape: (batch_size, 1)
        
        # Advantage stream - estimates A(s,a)
        advantage = F.relu(self.advantage_fc1(fused_features))
        advantage = self.dropout(advantage)
        advantage = self.advantage_fc2(advantage)  # Shape: (batch_size, num_actions)
        
        # Combine value and advantage using dueling architecture
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        
        return q_values
    
    def forward_dist(
        self,
        frames: torch.Tensor,
        state_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass returning C51 log-probability distributions.
        
        Only valid when ``num_atoms > 1``.
        
        Args:
            frames: Stacked frames tensor of shape (batch_size, 4, 84, 84)
            state_vector: Game state vector of shape (batch_size, 12)
            
        Returns:
            Log-probability distributions of shape (batch_size, num_actions, num_atoms)
        """
        assert self.distributional, "forward_dist() requires num_atoms > 1"
        
        batch_size = frames.size(0)
        N = self.num_atoms
        A = self.num_actions
        
        # Shared convolutional trunk
        x = F.relu(self.conv1(frames))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        conv_features = x.contiguous().view(batch_size, -1)
        
        # Fusion
        fused = torch.cat([conv_features, state_vector], dim=1)
        fused = F.relu(self.fusion_fc(fused))
        fused = self.dropout(fused)
        
        # Value stream -> (B, N)
        v = F.relu(self.value_fc1(fused))
        v = self.dropout(v)
        v = self.value_fc2(v)  # (B, N)
        v = v.view(batch_size, 1, N)
        
        # Advantage stream -> (B, A*N) -> (B, A, N)
        a = F.relu(self.advantage_fc1(fused))
        a = self.dropout(a)
        a = self.advantage_fc2(a)  # (B, A*N)
        a = a.view(batch_size, A, N)
        
        # Dueling combination per-atom: q(s,a,z) = v(s,z) + a(s,a,z) - mean_a(a(s,a,z))
        q_atoms = v + a - a.mean(dim=1, keepdim=True)
        
        # Softmax over atom dimension to get probabilities
        log_probs = F.log_softmax(q_atoms, dim=2)
        
        return log_probs
    
    def reset_noise(self):
        """Reset noise in all NoisyLinear layers. Call before each forward pass during training."""
        if not self.noisy:
            return
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    
    def get_action(
        self,
        frames: torch.Tensor,
        state_vector: torch.Tensor,
        epsilon: float = 0.0
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            frames: Stacked frames tensor
            state_vector: Game state vector
            epsilon: Exploration rate (0.0 for greedy)
            
        Returns:
            Selected action index
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.num_actions)
        
        with torch.no_grad():
            q_values = self.forward(frames, state_vector)
            return q_values.argmax(dim=1).item()
    
    def get_q_values(
        self,
        frames: torch.Tensor,
        state_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Get Q-values for given state.
        
        Args:
            frames: Stacked frames tensor
            state_vector: Game state vector
            
        Returns:
            Q-values tensor
        """
        with torch.no_grad():
            return self.forward(frames, state_vector)


class DuelingDQNConfig:
    """Configuration class for Dueling DQN model."""
    
    def __init__(self):
        # Network architecture
        self.num_actions = 12
        self.state_vector_size = 12
        self.frame_stack_size = 4
        self.frame_size = (84, 84)
        self.noisy = True              # Enable NoisyNet for exploration
        self.sigma_init = 0.5          # Initial noise magnitude
        
        # C51 distributional parameters (set num_atoms=1 to disable)
        self.num_atoms = 51            # Number of atoms (51 is the original C51 paper value)
        self.v_min = -10.0             # Minimum return value
        self.v_max = 10.0              # Maximum return value
        
        # Convolutional layers
        self.conv_layers = [
            {'filters': 32, 'kernel_size': 8, 'stride': 4, 'padding': 2},
            {'filters': 64, 'kernel_size': 4, 'stride': 2, 'padding': 1},
            {'filters': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ]
        
        # Fusion layer
        self.fusion_hidden_size = 512
        self.fusion_dropout = 0.3
        
        # Dueling streams
        self.value_hidden_size = 256
        self.advantage_hidden_size = 256
        self.stream_dropout = 0.3
        
        # Weight initialization
        self.init_method = 'kaiming_normal'
        self.init_nonlinearity = 'relu'
        self.bias_init = 0.0


def create_dueling_dqn(config: Optional[DuelingDQNConfig] = None) -> DuelingDQN:
    """
    Factory function to create a Dueling DQN model.
    
    Args:
        config: Optional configuration object
        
    Returns:
        Initialized Dueling DQN model
    """
    if config is None:
        config = DuelingDQNConfig()
    
    model = DuelingDQN(
        num_actions=config.num_actions,
        state_vector_size=config.state_vector_size,
        frame_stack_size=config.frame_stack_size,
        frame_size=config.frame_size,
        noisy=getattr(config, 'noisy', True),
        sigma_init=getattr(config, 'sigma_init', 0.5),
        num_atoms=getattr(config, 'num_atoms', 1),
        v_min=getattr(config, 'v_min', -10.0),
        v_max=getattr(config, 'v_max', 10.0)
    )
    
    return model


# Action space mapping for reference
ACTION_SPACE = {
    0: "no_action",
    1: "right",
    2: "left",
    3: "jump",
    4: "right_jump",
    5: "left_jump",
    6: "run",
    7: "right_run",
    8: "left_run",
    9: "right_jump_run",  # Forward jumping
    10: "left_jump_run",
    11: "crouch"
}


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_dueling_dqn().to(device)
    
    # Test forward pass
    batch_size = 4
    frames = torch.randn(batch_size, 4, 84, 84).to(device)
    state_vector = torch.randn(batch_size, 12).to(device)
    
    q_values = model(frames, state_vector)
    print(f"Model output shape: {q_values.shape}")
    print(f"Q-values sample: {q_values[0].detach().cpu().numpy()}")
    
    # Test action selection
    action = model.get_action(frames[:1], state_vector[:1], epsilon=0.1)
    print(f"Selected action: {action} ({ACTION_SPACE[action]})")
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")