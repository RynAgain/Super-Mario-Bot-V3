"""
Environment module for Super Mario Bros AI training system.

This module handles reward calculation, episode management, and game state tracking
for the reinforcement learning environment.
"""

from .reward_calculator import RewardCalculator
from .episode_manager import EpisodeManager

__all__ = ['RewardCalculator', 'EpisodeManager']