"""
Neural Network Models for Super Mario Bros AI

This module contains neural network architectures for the Mario AI training system.

Available Models:
- DuelingDQN: Dueling Deep Q-Network with 4-frame stacking
"""

from python.models.dueling_dqn import DuelingDQN, create_dueling_dqn, DuelingDQNConfig, ACTION_SPACE

__all__ = [
    'DuelingDQN',
    'create_dueling_dqn',
    'DuelingDQNConfig',
    'ACTION_SPACE'
]