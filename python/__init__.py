"""
Super Mario Bros AI Training System

This package contains the neural network components for training an AI agent
to play Super Mario Bros using Deep Q-Learning with Dueling DQN architecture.

Modules:
- models: Neural network architectures (Dueling DQN)
- agents: DQN training agents
- utils: Utilities for preprocessing, replay buffer, model management, etc.
"""

__version__ = "1.0.0"
__author__ = "Super Mario Bot V3 Team"
__description__ = "Neural network components for Super Mario Bros AI training"

# Import main components for easy access
from python.models.dueling_dqn import DuelingDQN, create_dueling_dqn
from python.agents.dqn_agent import DQNAgent
from python.utils.config_loader import ConfigLoader, load_config

__all__ = [
    'DuelingDQN',
    'create_dueling_dqn', 
    'DQNAgent',
    'ConfigLoader',
    'load_config'
]