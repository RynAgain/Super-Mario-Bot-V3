"""
Communication module for Super Mario Bros AI training system.

This module handles WebSocket communication between the FCEUX Lua script
and the Python AI trainer, including binary protocol parsing and message routing.
"""

from .websocket_server import WebSocketServer
from .comm_manager import CommunicationManager

__all__ = ['WebSocketServer', 'CommunicationManager']