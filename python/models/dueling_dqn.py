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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class DuelingDQN(nn.Module):
    """
    Dueling DQN implementation with 4-frame stacking and game state fusion.
    
    The network processes stacked frames through convolutional layers,
    fuses them with game state features, and uses dueling architecture
    to separate value and advantage estimation.
    """
    
    def __init__(
        self,
        num_actions: int = 12,
        state_vector_size: int = 12,
        frame_stack_size: int = 4,
        frame_size: Tuple[int, int] = (84, 84)
    ):
        """
        Initialize the Dueling DQN model.
        
        Args:
            num_actions: Number of possible actions (12 for Mario)
            state_vector_size: Size of game state vector (12 features)
            frame_stack_size: Number of frames to stack (4 frames)
            frame_size: Frame dimensions (height, width) = (84, 84)
        """
        super(DuelingDQN, self).__init__()
        
        self.num_actions = num_actions
        self.state_vector_size = state_vector_size
        self.frame_stack_size = frame_stack_size
        self.frame_size = frame_size
        
        # Convolutional layers for frame processing
        self.conv1 = nn.Conv2d(
            in_channels=frame_stack_size,  # 4 stacked frames
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
        
        # Feature fusion layer
        self.fusion_fc = nn.Linear(
            self.conv_output_size + state_vector_size,
            512
        )
        
        # Value stream (estimates state value)
        self.value_fc1 = nn.Linear(512, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Advantage stream (estimates action advantages)
        self.advantage_fc1 = nn.Linear(512, 256)
        self.advantage_fc2 = nn.Linear(256, num_actions)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
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
            Q-values tensor of shape (batch_size, num_actions)
        """
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
        frame_size=config.frame_size
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