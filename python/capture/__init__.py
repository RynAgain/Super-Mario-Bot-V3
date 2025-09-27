"""
Frame capture module for Super Mario Bros AI training system.

This module handles window capture from FCEUX using cv2, frame synchronization,
and preprocessing for neural network input.
"""

from .frame_capture import FrameCapture

__all__ = ['FrameCapture']