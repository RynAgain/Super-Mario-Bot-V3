"""
AI Agents for Super Mario Bros Training

This module contains the DQN agent implementation for training the Mario AI.

Available Agents:
- DQNAgent: Deep Q-Network agent with Dueling DQN, experience replay, and target networks
"""

from python.agents.dqn_agent import DQNAgent

__all__ = [
    'DQNAgent'
]