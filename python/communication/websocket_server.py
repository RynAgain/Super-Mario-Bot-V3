"""
WebSocket server for Super Mario Bros AI training system.

Handles WebSocket communication between FCEUX Lua script and Python trainer,
implementing the hybrid JSON/binary protocol for control messages and game state data.
"""

import asyncio
import json
import logging
import struct
import time
from typing import Dict, Any, Optional, Callable, List
import websockets
from websockets.server import WebSocketServerProtocol


class WebSocketServer:
    """
    WebSocket server that handles communication with FCEUX Lua script.
    
    Implements the hybrid protocol:
    - JSON messages for control, configuration, and debug information
    - Binary messages for high-frequency game state data
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Initialize WebSocket server.
        
        Args:
            host: Server host address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.server = None
        self.client_websocket: Optional[WebSocketServerProtocol] = None
        self.is_running = False
        self.protocol_version = "1.0"
        
        # Message handlers
        self.json_handlers: Dict[str, Callable] = {}
        self.binary_handler: Optional[Callable] = None
        
        # Connection state
        self.client_info: Dict[str, Any] = {}
        self.last_ping_time = 0
        self.ping_interval = 1.0  # 1 second
        
        # Frame synchronization
        self.current_frame_id = 0
        self.frame_sync_enabled = True
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default message handlers."""
        self.json_handlers.update({
            'init': self._handle_init,
            'ping': self._handle_ping,
            'pong': self._handle_pong,
            'episode_event': self._handle_episode_event,
            'frame_advance': self._handle_frame_advance,
            'error': self._handle_error
        })
    
    def register_json_handler(self, message_type: str, handler: Callable):
        """
        Register a handler for JSON messages.
        
        Args:
            message_type: Type of message to handle
            handler: Async function to handle the message
        """
        self.json_handlers[message_type] = handler
    
    def register_binary_handler(self, handler: Callable):
        """
        Register a handler for binary game state messages.
        
        Args:
            handler: Async function to handle binary data
        """
        self.binary_handler = handler
    
    async def start_server(self):
        """Start the WebSocket server."""
        try:
            self.logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
            self.server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                max_size=65536,  # 64KB buffer
                compression=None  # Disable compression for latency
            )
            self.is_running = True
            self.logger.info("WebSocket server started successfully")
            
            # Start ping task
            asyncio.create_task(self._ping_task())
            
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
            raise
    
    async def stop_server(self):
        """Stop the WebSocket server."""
        if self.server:
            self.logger.info("Stopping WebSocket server")
            self.is_running = False
            self.server.close()
            await self.server.wait_closed()
            self.client_websocket = None
            self.logger.info("WebSocket server stopped")
    
    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """
        Handle client connection.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        client_address = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.logger.info(f"Client connected from {client_address}")
        
        # Store client connection
        self.client_websocket = websocket
        
        try:
            async for message in websocket:
                await self._process_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client {client_address} disconnected")
        except Exception as e:
            self.logger.error(f"Error handling client {client_address}: {e}")
            await self._send_error("INTERNAL_ERROR", str(e))
        finally:
            self.client_websocket = None
            self.client_info.clear()
    
    async def _process_message(self, message):
        """
        Process incoming message (JSON or binary).
        
        Args:
            message: Raw message data
        """
        try:
            if isinstance(message, str):
                # JSON message
                await self._process_json_message(message)
            elif isinstance(message, bytes):
                # Binary message
                await self._process_binary_message(message)
            else:
                await self._send_error("INVALID_MESSAGE", "Unknown message format")
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            await self._send_error("MESSAGE_PROCESSING_ERROR", str(e))
    
    async def _process_json_message(self, message: str):
        """
        Process JSON control message.
        
        Args:
            message: JSON message string
        """
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if not message_type:
                await self._send_error("INVALID_MESSAGE", "Missing message type")
                return
            
            # Find and call handler
            handler = self.json_handlers.get(message_type)
            if handler:
                await handler(data)
            else:
                self.logger.warning(f"No handler for message type: {message_type}")
                await self._send_error("UNKNOWN_MESSAGE_TYPE", f"No handler for {message_type}")
                
        except json.JSONDecodeError as e:
            await self._send_error("INVALID_JSON", f"JSON decode error: {e}")
        except Exception as e:
            await self._send_error("JSON_PROCESSING_ERROR", str(e))
    
    async def _process_binary_message(self, message: bytes):
        """
        Process binary game state message.
        
        Args:
            message: Binary message data
        """
        try:
            # Validate message structure according to protocol
            if len(message) < 8:
                await self._send_error("INVALID_BINARY", "Message too short for header")
                return
            
            # Parse header (8 bytes)
            header = struct.unpack('<BIHB', message[:8])
            msg_type, frame_id, data_length, checksum = header
            
            # Validate message type
            if msg_type != 0x01:  # game_state
                await self._send_error("INVALID_BINARY", f"Unknown binary message type: {msg_type}")
                return
            
            # Validate data length
            if len(message) != 8 + data_length:
                await self._send_error("INVALID_BINARY", "Data length mismatch")
                return
            
            # Validate checksum
            payload = message[8:]
            calculated_checksum = self._calculate_checksum(payload)
            if calculated_checksum != checksum:
                await self._send_error("INVALID_BINARY", "Checksum mismatch")
                return
            
            # Update frame tracking
            self.current_frame_id = frame_id
            
            # Call binary handler if registered
            if self.binary_handler:
                await self.binary_handler(frame_id, payload)
            else:
                self.logger.warning("Received binary data but no handler registered")
                
        except struct.error as e:
            await self._send_error("BINARY_PARSE_ERROR", f"Failed to parse binary data: {e}")
        except Exception as e:
            await self._send_error("BINARY_PROCESSING_ERROR", str(e))
    
    def _calculate_checksum(self, data: bytes) -> int:
        """
        Calculate simple XOR checksum.
        
        Args:
            data: Data to checksum
            
        Returns:
            XOR checksum value
        """
        checksum = 0
        for byte in data:
            checksum ^= byte
        return checksum & 0xFF
    
    # Default message handlers
    
    async def _handle_init(self, data: Dict[str, Any]):
        """Handle initialization message from Lua script."""
        self.logger.info("Received initialization message")
        
        # Store client info
        self.client_info = {
            'fceux_version': data.get('fceux_version'),
            'rom_name': data.get('rom_name'),
            'protocol_version': data.get('protocol_version'),
            'connected_at': time.time()
        }
        
        # Validate protocol version
        if data.get('protocol_version') != self.protocol_version:
            await self._send_error("PROTOCOL_VERSION_MISMATCH", 
                                 f"Expected {self.protocol_version}, got {data.get('protocol_version')}")
            return
        
        # Send acknowledgment
        response = {
            'type': 'init_ack',
            'timestamp': int(time.time() * 1000),
            'status': 'ready',
            'training_mode': 'world_1_1',
            'frame_skip': 4
        }
        
        await self._send_json(response)
        self.logger.info("Sent initialization acknowledgment")
    
    async def _handle_ping(self, data: Dict[str, Any]):
        """Handle ping message."""
        timestamp = data.get('timestamp', int(time.time() * 1000))
        current_time = int(time.time() * 1000)
        latency = current_time - timestamp
        
        response = {
            'type': 'pong',
            'timestamp': current_time,
            'latency_ms': latency
        }
        
        await self._send_json(response)
    
    async def _handle_pong(self, data: Dict[str, Any]):
        """Handle pong response."""
        latency = data.get('latency_ms', 0)
        self.logger.debug(f"Received pong with latency: {latency}ms")
    
    async def _handle_episode_event(self, data: Dict[str, Any]):
        """Handle episode event message."""
        event = data.get('event')
        episode_id = data.get('episode_id')
        self.logger.info(f"Episode {episode_id} event: {event}")
        
        # This will be handled by the episode manager
        # For now, just log the event
    
    async def _handle_frame_advance(self, data: Dict[str, Any]):
        """Handle frame advance notification."""
        frame_id = data.get('frame_id')
        
        if self.frame_sync_enabled:
            # Send frame acknowledgment
            response = {
                'type': 'frame_ack',
                'frame_id': frame_id,
                'ready_for_next': True
            }
            await self._send_json(response)
    
    async def _handle_error(self, data: Dict[str, Any]):
        """Handle error message from client."""
        error_code = data.get('error_code')
        message = data.get('message')
        self.logger.error(f"Client error [{error_code}]: {message}")
    
    # Message sending methods
    
    async def _send_json(self, data: Dict[str, Any]):
        """
        Send JSON message to client.
        
        Args:
            data: Data to send as JSON
        """
        if not self.client_websocket:
            self.logger.warning("No client connected, cannot send JSON message")
            return
        
        try:
            message = json.dumps(data, separators=(',', ':'))  # Compact format
            await self.client_websocket.send(message)
        except Exception as e:
            self.logger.error(f"Failed to send JSON message: {e}")
    
    async def _send_error(self, error_code: str, message: str):
        """
        Send error message to client.
        
        Args:
            error_code: Error code
            message: Error message
        """
        error_data = {
            'type': 'error',
            'error_code': error_code,
            'message': message,
            'timestamp': int(time.time() * 1000)
        }
        
        await self._send_json(error_data)
    
    async def send_action(self, action_buttons: Dict[str, bool], frame_id: Optional[int] = None):
        """
        Send action command to Lua script.
        
        Args:
            action_buttons: Button states dictionary
            frame_id: Frame ID for synchronization
        """
        if frame_id is None:
            frame_id = self.current_frame_id
        
        action_data = {
            'type': 'action',
            'frame_id': frame_id,
            'buttons': action_buttons,
            'hold_frames': 1
        }
        
        await self._send_json(action_data)
    
    async def send_training_control(self, command: str, episode_id: Optional[int] = None, 
                                  reset_to_level: str = "1-1"):
        """
        Send training control command.
        
        Args:
            command: Control command (start, pause, reset, stop)
            episode_id: Episode ID
            reset_to_level: Level to reset to
        """
        control_data = {
            'type': 'training_control',
            'command': command,
            'episode_id': episode_id,
            'reset_to_level': reset_to_level
        }
        
        await self._send_json(control_data)
    
    async def _ping_task(self):
        """Background task to send periodic pings."""
        while self.is_running:
            try:
                await asyncio.sleep(self.ping_interval)
                
                if self.client_websocket and time.time() - self.last_ping_time > self.ping_interval:
                    ping_data = {
                        'type': 'ping',
                        'timestamp': int(time.time() * 1000)
                    }
                    await self._send_json(ping_data)
                    self.last_ping_time = time.time()
                    
            except Exception as e:
                self.logger.error(f"Error in ping task: {e}")
    
    # Utility methods
    
    def is_client_connected(self) -> bool:
        """Check if client is connected."""
        return self.client_websocket is not None
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get client connection information."""
        return self.client_info.copy()
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        if not self.client_websocket:
            return {'connected': False}
        
        return {
            'connected': True,
            'client_address': f"{self.client_websocket.remote_address[0]}:{self.client_websocket.remote_address[1]}",
            'protocol_version': self.client_info.get('protocol_version'),
            'fceux_version': self.client_info.get('fceux_version'),
            'rom_name': self.client_info.get('rom_name'),
            'connected_duration': time.time() - self.client_info.get('connected_at', time.time()),
            'current_frame_id': self.current_frame_id
        }