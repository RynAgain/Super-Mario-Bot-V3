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
import random
from typing import Dict, Any, Optional, Callable, List, Set
import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK

from ..utils.preprocessing import BinaryPayloadParser, EnhancedFeatureValidator


class ConnectionState:
    """Tracks connection state and health metrics."""
    
    def __init__(self):
        self.connected = False
        self.connection_start_time = 0
        self.last_ping_sent = 0
        self.last_pong_received = 0
        self.last_message_received = 0
        self.ping_failures = 0
        self.consecutive_timeouts = 0
        self.total_messages_sent = 0
        self.total_messages_received = 0
        self.reconnect_attempts = 0
        self.connection_quality = 1.0  # 0.0 to 1.0
        self.latency_ms = 0
        self.is_degraded = False


class WebSocketServer:
    """
    WebSocket server that handles communication with FCEUX Lua script.
    
    Implements the hybrid protocol with robust connection handling:
    - JSON messages for control, configuration, and debug information
    - Binary messages for high-frequency game state data
    - Proper ping/pong keepalive mechanism
    - Connection health monitoring and graceful degradation
    - Exponential backoff reconnection with jitter
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765, enhanced_features: bool = False):
        """
        Initialize WebSocket server.
        
        Args:
            host: Server host address
            port: Server port number
            enhanced_features: Whether to enable enhanced 20-feature mode
        """
        self.host = host
        self.port = port
        self.server = None
        self.client_websocket: Optional[WebSocketServerProtocol] = None
        self.is_running = False
        self.protocol_version = "1.1"  # Updated for enhanced features support
        
        # Enhanced features configuration
        self.enhanced_features = enhanced_features
        self.binary_parser = BinaryPayloadParser(enhanced_features)
        self.feature_validator = EnhancedFeatureValidator(enhanced_features)
        
        # Message handlers
        self.json_handlers: Dict[str, Callable] = {}
        self.binary_handler: Optional[Callable] = None
        self.enhanced_state_handler: Optional[Callable] = None
        
        # Connection state and health monitoring
        self.connection_state = ConnectionState()
        self.client_info: Dict[str, Any] = {}
        
        # Heartbeat and keepalive configuration
        self.ping_interval = 15.0  # Send ping every 15 seconds
        self.ping_timeout = 45.0   # Timeout after 45 seconds without pong
        self.heartbeat_interval = 5.0  # Send heartbeat message every 5 seconds
        self.last_heartbeat_sent = 0
        self.pending_pings: Set[str] = set()  # Track pending ping IDs
        
        # Connection quality and degradation
        self.quality_check_interval = 10.0  # Check quality every 10 seconds
        self.last_quality_check = 0
        self.degradation_threshold = 0.7  # Quality below this triggers degradation
        self.recovery_threshold = 0.9     # Quality above this recovers from degradation
        
        # Reconnection with exponential backoff
        self.base_reconnect_delay = 1.0    # Base delay in seconds
        self.max_reconnect_delay = 30.0    # Maximum delay in seconds
        self.reconnect_jitter = 0.1        # Jitter factor (10%)
        self.max_reconnect_attempts = 10   # Maximum reconnection attempts
        
        # Frame synchronization
        self.current_frame_id = 0
        self.frame_sync_enabled = True
        self.frame_ack_enabled = True  # Re-enable frame acknowledgments
        
        # Input queuing during reconnection
        self.input_queue = asyncio.Queue(maxsize=100)
        self.is_reconnecting = False
        
        # Enhanced communication statistics
        self.enhanced_stats = {
            'total_enhanced_frames': 0,
            'successful_parses': 0,
            'validation_errors': 0,
            'feature_validation_warnings': 0,
            'payload_size_mismatches': 0
        }
        
        # Connection quality metrics
        self.quality_metrics = {
            'ping_success_rate': 1.0,
            'message_success_rate': 1.0,
            'average_latency': 0.0,
            'connection_uptime': 0.0,
            'reconnection_count': 0,
            'degradation_events': 0
        }
        
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
            'error': self._handle_error,
            'reset_request': self._handle_reset_request,
            'resync_request': self._handle_resync_request,
            'disconnect': self._handle_disconnect
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
    
    def register_enhanced_state_handler(self, handler: Callable):
        """
        Register a handler for enhanced parsed game state.
        
        Args:
            handler: Async function to handle parsed game state dictionary
        """
        self.enhanced_state_handler = handler
    
    def set_enhanced_features(self, enabled: bool):
        """
        Enable or disable enhanced features mode.
        
        Args:
            enabled: Whether to enable enhanced features
        """
        self.enhanced_features = enabled
        self.binary_parser = BinaryPayloadParser(enabled)
        self.feature_validator = EnhancedFeatureValidator(enabled)
        self.logger.info(f"Enhanced features {'enabled' if enabled else 'disabled'}")
    
    async def start_server(self):
        """Start the WebSocket server."""
        try:
            self.logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
            
            # Create a wrapper function to handle both old and new websockets API
            async def handler_wrapper(websocket, path=None):
                return await self._handle_client(websocket, path or "/")
            
            self.server = await websockets.serve(
                handler_wrapper,
                self.host,
                self.port,
                max_size=65536,  # 64KB buffer
                max_queue=1,     # Back-pressure: limit queued messages
                compression=None,  # Disable compression for latency
                ping_interval=10.0,  # more frequent heartbeats
                ping_timeout=10.0    # Close connection if ping not responded in 10 seconds
            )
            self.is_running = True
            self.logger.info("WebSocket server started successfully")
            
            # Start ping monitoring task
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
    
    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str = "/"):
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
                
        except websockets.exceptions.ConnectionClosed as e:
            close_code = getattr(e, 'code', '?')
            close_reason = getattr(e, 'reason', '')
            self.logger.info(f"Client {client_address} disconnected (code={close_code}, reason='{close_reason}')")
            if close_code in [1006, 1011]:
                self.logger.warning("Close code suggests ping/pong timeout or abnormal closure")
        except Exception as e:
            self.logger.error(f"Error handling client {client_address}: {e}")
            # Don't send error responses during connection handling errors
            # Just log and let the connection close naturally
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
            # Don't send error response for processing errors - just log and continue
    
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
            self.logger.warning(f"Invalid JSON received: {e}")
            # Don't send error response - just log and continue
        except Exception as e:
            self.logger.error(f"Error processing JSON message: {e}")
            # Don't send error response - just log and continue
    
    async def _process_binary_message(self, message: bytes):
        """
        Process binary game state message with enhanced payload parsing and validation.
        
        Args:
            message: Binary message data
        """
        try:
            # Validate message size limits
            if len(message) > 65536:  # 64KB limit
                self.logger.warning(f"Binary message too large: {len(message)} bytes, dropping frame")
                return
            
            # Validate minimum message structure
            if len(message) < 8:
                self.logger.warning(f"Binary message too short: {len(message)} bytes, expected at least 8 bytes for header")
                return
            
            # Parse header (8 bytes) with error handling
            try:
                header = struct.unpack('<BIHB', message[:8])
                msg_type, frame_id, data_length, checksum = header
            except struct.error as e:
                self.logger.warning(f"Failed to parse binary message header: {e}, dropping frame")
                return
            
            # Validate message type
            if msg_type != 0x01:  # game_state
                self.logger.warning(f"Unknown binary message type: {msg_type}, dropping frame")
                return
            
            # Validate payload length
            actual_payload_len = len(message) - 8
            if actual_payload_len <= 0:
                self.logger.warning("Binary message has no payload, dropping frame")
                return
            
            # Log payload size mismatches for debugging
            if data_length != actual_payload_len:
                self.logger.debug(f"Payload size mismatch: header={data_length}, actual={actual_payload_len}")
                self.enhanced_stats['payload_size_mismatches'] += 1
            
            # Enhanced payload validation - expect 128 bytes for enhanced mode
            expected_payload_size = 128 if self.enhanced_features else 80  # Backward compatibility
            if actual_payload_len < 16:
                self.logger.warning(f"Payload too small: {actual_payload_len} bytes, expected at least 16 bytes")
                return
            elif actual_payload_len > 256:
                self.logger.warning(f"Payload too large: {actual_payload_len} bytes, expected max 256 bytes")
                return
            elif self.enhanced_features and actual_payload_len != 128:
                self.logger.debug(f"Enhanced mode expects 128-byte payload, got {actual_payload_len} bytes")
            
            # Extract and validate checksum
            payload = message[8:]
            calculated_checksum = self._calculate_checksum(payload)
            if calculated_checksum != checksum:
                self.logger.warning(f"Checksum mismatch: expected {checksum}, got {calculated_checksum}, dropping frame")
                return
            
            # Update frame tracking and statistics
            self.current_frame_id = frame_id
            self.enhanced_stats['total_enhanced_frames'] += 1
            
            # Enhanced payload processing
            parsed_game_state = None
            if self.enhanced_features and actual_payload_len == 128:
                try:
                    # Validate binary payload structure
                    validation_result = self.feature_validator.validate_binary_payload(payload)
                    if not validation_result['valid']:
                        self.logger.warning(f"Binary payload validation failed: {validation_result['errors']}")
                        self.enhanced_stats['validation_errors'] += 1
                        # Continue with basic processing
                    elif validation_result['warnings']:
                        self.logger.debug(f"Binary payload warnings: {validation_result['warnings']}")
                        self.enhanced_stats['feature_validation_warnings'] += 1
                    
                    # Parse enhanced payload
                    parsed_game_state = self.binary_parser.parse_payload(payload)
                    self.enhanced_stats['successful_parses'] += 1
                    
                    # Validate parsed state
                    state_validation = self.feature_validator.validate_game_state(parsed_game_state)
                    if state_validation['warnings']:
                        self.logger.debug(f"Game state validation warnings: {state_validation['warnings']}")
                    
                    self.logger.debug(f"Enhanced payload parsed successfully for frame {frame_id}")
                    
                except Exception as parse_error:
                    self.logger.warning(f"Enhanced payload parsing failed for frame {frame_id}: {parse_error}")
                    self.enhanced_stats['validation_errors'] += 1
                    parsed_game_state = None
            
            self.logger.debug(f"Processing binary frame {frame_id} with {actual_payload_len} bytes payload")
            
            # Call enhanced state handler if available and parsing succeeded
            if self.enhanced_state_handler and parsed_game_state:
                try:
                    await self.enhanced_state_handler(frame_id, parsed_game_state)
                except Exception as handler_error:
                    self.logger.error(f"Enhanced state handler failed for frame {frame_id}: {handler_error}")
                    # Continue with basic handler
            
            # Call binary handler if registered with error isolation
            if self.binary_handler:
                try:
                    await self.binary_handler(frame_id, payload)
                except Exception as handler_error:
                    self.logger.error(f"Binary handler failed for frame {frame_id}: {handler_error}")
                    # Don't close connection for handler errors - just log and continue
            else:
                self.logger.warning("Received binary data but no handler registered")
                
        except Exception as e:
            self.logger.error(f"Unexpected error processing binary message: {e}")
            self.enhanced_stats['validation_errors'] += 1
            # Don't send error response or close connection - just log and continue
    
    def _calculate_checksum(self, data: bytes) -> int:
        """
        Calculate simple sum checksum to match Lua script.
        
        Args:
            data: Data to checksum
            
        Returns:
            Sum checksum value (modulo 256)
        """
        checksum = 0
        for byte in data:
            checksum = (checksum + byte) % 256
        return checksum
    
    # Default message handlers
    
    async def _handle_init(self, data: Dict[str, Any]):
        """Handle initialization message from Lua script."""
        self.logger.info("Received initialization message")
        
        # Store client info
        self.client_info = {
            'fceux_version': data.get('fceux_version'),
            'rom_name': data.get('rom_name'),
            'protocol_version': data.get('protocol_version'),
            'connected_at': time.time(),
            'enhanced_features_supported': data.get('enhanced_features_supported', False)
        }
        
        # Validate protocol version (allow both 1.0 and 1.1 for backward compatibility)
        client_version = data.get('protocol_version', '1.0')
        if client_version not in ['1.0', '1.1']:
            await self._send_error("PROTOCOL_VERSION_MISMATCH",
                                 f"Expected 1.0 or 1.1, got {client_version}")
            return
        
        # Check enhanced features compatibility
        client_enhanced_support = data.get('enhanced_features_supported', False)
        if self.enhanced_features and not client_enhanced_support:
            self.logger.warning("Server has enhanced features enabled but client doesn't support them")
            # Don't fail - fall back to basic mode
            self.set_enhanced_features(False)
        
        # Send acknowledgment with enhanced features information
        response = {
            'type': 'init_ack',
            'timestamp': int(time.time() * 1000),
            'status': 'ready',
            'training_mode': 'world_1_1',
            'frame_skip': 4,
            'protocol_version': self.protocol_version,
            'enhanced_features_enabled': self.enhanced_features,
            'expected_payload_size': 128 if self.enhanced_features else 80,
            'state_vector_size': 20 if self.enhanced_features else 12
        }
        
        await self._send_json(response)
        self.logger.info(f"Sent initialization acknowledgment (enhanced_features: {self.enhanced_features})")
    
    async def _handle_ping(self, data: Dict[str, Any]):
        """Handle ping message."""
        timestamp = data.get('timestamp', int(time.time() * 1000))
        current_time = int(time.time() * 1000)
        latency = current_time - timestamp
        
        # Disable pong responses to prevent potential frame corruption
        # response = {
        #     'type': 'pong',
        #     'timestamp': current_time,
        #     'latency_ms': latency
        # }
        #
        # success = await self._send_json(response)
        # if not success:
        #     self.logger.warning("Failed to send pong response")
        
        self.logger.debug(f"Received ping with latency: {latency}ms (pong disabled)")
    
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
        
        # Disable frame acknowledgments to prevent potential frame corruption
        # if self.frame_sync_enabled:
        #     # Send frame acknowledgment
        #     response = {
        #         'type': 'frame_ack',
        #         'frame_id': frame_id,
        #         'ready_for_next': True
        #     }
        #     await self._send_json(response)
    
    async def _handle_error(self, data: Dict[str, Any]):
        """Handle error message from client."""
        error_code = data.get('error_code')
        message = data.get('message')
        self.logger.error(f"Client error [{error_code}]: {message}")
    
    async def _handle_reset_request(self, data: Dict[str, Any]):
        """Handle reset request from Lua script."""
        episode_id = data.get('episode_id')
        self.logger.info(f"Received reset request for episode {episode_id}")
        
        # Send reset acknowledgment
        response = {
            'type': 'reset_ack',
            'episode_id': episode_id,
            'timestamp': int(time.time() * 1000),
            'status': 'acknowledged'
        }
        
        await self._send_json(response)
        self.logger.info("Reset request acknowledged - manual reset required in FCEUX")
    
    async def _handle_resync_request(self, data: Dict[str, Any]):
        """Handle resync request from Lua script."""
        current_frame = data.get('current_frame', 0)
        self.logger.info(f"Received resync request for frame {current_frame}")
        
        # Reset frame synchronization
        self.current_frame_id = current_frame
        
        # Send resync acknowledgment
        response = {
            'type': 'resync_ack',
            'frame_id': current_frame,
            'timestamp': int(time.time() * 1000),
            'status': 'synchronized'
        }
        
        await self._send_json(response)
        self.logger.info(f"Resync acknowledged - synchronized to frame {current_frame}")
    
    async def _handle_disconnect(self, data: Dict[str, Any]):
        """Handle disconnect message from Lua script."""
        reason = data.get('reason', 'unknown')
        self.logger.info(f"Client disconnect notification: {reason}")
        # Just acknowledge the disconnect, connection will close naturally
    
    # Message sending methods
    
    async def _send_json(self, data: Dict[str, Any]):
        """
        Send JSON message to client.
        
        Args:
            data: Data to send as JSON
        """
        if not self.client_websocket:
            self.logger.warning("No client connected, cannot send JSON message")
            return False
        
        # Check if connection is closed using the correct method
        try:
            if hasattr(self.client_websocket, 'closed') and self.client_websocket.closed:
                self.logger.warning("Connection closed, cannot send JSON message")
                return False
        except AttributeError:
            # For older websockets versions, we'll just try to send and handle the exception
            pass
        
        try:
            message = json.dumps(data, separators=(',', ':'))  # Compact format
            await self.client_websocket.send(message)
            return True
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("Connection closed while sending JSON message")
            self.client_websocket = None
            return False
        except Exception as e:
            self.logger.error(f"Failed to send JSON message: {e}")
            return False
    
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
        
        success = await self._send_json(action_data)
        if not success:
            self.logger.warning(f"Failed to send action for frame {frame_id}")
        return success
    
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
        
        success = await self._send_json(control_data)
        if not success:
            self.logger.warning(f"Failed to send training control command: {command}")
        return success
    
    async def _ping_task(self):
        """Background task to monitor connection health and send pings."""
        while self.is_running:
            try:
                await asyncio.sleep(20.0)  # Check every 20 seconds
                
                if self.client_websocket and not getattr(self.client_websocket, 'closed', True):
                    try:
                        # Send ping and wait for pong with timeout
                        pong_waiter = await self.client_websocket.ping()
                        await asyncio.wait_for(pong_waiter, timeout=10.0)
                        self.logger.debug("Ping successful - connection healthy")
                        
                    except asyncio.TimeoutError:
                        self.logger.warning("Ping timeout - closing connection")
                        try:
                            await self.client_websocket.close(code=1011, reason="Ping timeout")
                        except Exception:
                            pass  # Connection might already be closed
                        self.client_websocket = None
                        
                    except Exception as e:
                        self.logger.warning(f"Ping failed: {e} - closing connection")
                        try:
                            await self.client_websocket.close(code=1011, reason="Ping failed")
                        except Exception:
                            pass  # Connection might already be closed
                        self.client_websocket = None
                    
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
        
        stats = {
            'connected': True,
            'client_address': f"{self.client_websocket.remote_address[0]}:{self.client_websocket.remote_address[1]}",
            'protocol_version': self.client_info.get('protocol_version'),
            'fceux_version': self.client_info.get('fceux_version'),
            'rom_name': self.client_info.get('rom_name'),
            'connected_duration': time.time() - self.client_info.get('connected_at', time.time()),
            'current_frame_id': self.current_frame_id,
            'enhanced_features_enabled': self.enhanced_features,
            'client_enhanced_support': self.client_info.get('enhanced_features_supported', False)
        }
        
        # Add enhanced statistics if enabled
        if self.enhanced_features:
            stats.update(self.get_enhanced_stats())
        
        return stats
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced communication statistics."""
        stats = self.enhanced_stats.copy()
        
        # Calculate success rates
        total_frames = stats['total_enhanced_frames']
        if total_frames > 0:
            stats['parse_success_rate'] = stats['successful_parses'] / total_frames
            stats['validation_error_rate'] = stats['validation_errors'] / total_frames
            stats['warning_rate'] = stats['feature_validation_warnings'] / total_frames
        else:
            stats['parse_success_rate'] = 0.0
            stats['validation_error_rate'] = 0.0
            stats['warning_rate'] = 0.0
        
        return stats
    
    def reset_enhanced_stats(self):
        """Reset enhanced communication statistics."""
        self.enhanced_stats = {
            'total_enhanced_frames': 0,
            'successful_parses': 0,
            'validation_errors': 0,
            'feature_validation_warnings': 0,
            'payload_size_mismatches': 0
        }
        self.logger.info("Enhanced communication statistics reset")