#!/usr/bin/env python3
"""
Communication Test Script for Super Mario Bros AI Training System

This script tests the WebSocket communication between Python trainer and Lua script
without running the full training system. It helps diagnose connection issues.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from python.communication.websocket_server import WebSocketServer

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/communication_test.log')
    ]
)

logger = logging.getLogger(__name__)

class CommunicationTester:
    """Test WebSocket communication with Lua script."""
    
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.websocket_server = WebSocketServer(host, port)
        self.game_state_count = 0
        self.action_count = 0
        self.test_running = False
        
        # Register handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register WebSocket message handlers."""
        self.websocket_server.register_binary_handler(self._handle_game_state)
        self.websocket_server.register_json_handler('init', self._handle_init)
        self.websocket_server.register_json_handler('ping', self._handle_ping)
        self.websocket_server.register_json_handler('episode_event', self._handle_episode_event)
        self.websocket_server.register_json_handler('frame_advance', self._handle_frame_advance)
        self.websocket_server.register_json_handler('error', self._handle_error)
    
    async def _handle_game_state(self, frame_id: int, game_state_data: bytes):
        """Handle incoming game state data."""
        self.game_state_count += 1
        
        logger.info(f"Received game state #{self.game_state_count}, frame_id={frame_id}, size={len(game_state_data)} bytes")
        
        # Parse some basic info from binary data if possible
        if len(game_state_data) >= 16:  # Mario data block
            mario_x = int.from_bytes(game_state_data[0:2], byteorder='little')
            mario_y = int.from_bytes(game_state_data[2:4], byteorder='little')
            logger.info(f"  Mario position: X={mario_x}, Y={mario_y}")
        
        # Send a test action back
        await self._send_test_action(frame_id)
    
    async def _send_test_action(self, frame_id: int):
        """Send a test action to Lua script."""
        self.action_count += 1
        
        # Cycle through different actions for testing
        action_id = self.action_count % 12
        
        # Create action buttons based on action ID
        action_buttons = self._action_id_to_buttons(action_id)
        
        logger.info(f"Sending action #{self.action_count}: action_id={action_id}, buttons={action_buttons}")
        
        await self.websocket_server.send_action(action_buttons, frame_id)
    
    def _action_id_to_buttons(self, action_id: int) -> dict:
        """Convert action ID to button mapping."""
        action_map = {
            0: {},  # No action
            1: {'right': True},  # Move right
            2: {'right': True, 'A': True},  # Run right
            3: {'right': True, 'B': True},  # Jump right
            4: {'right': True, 'A': True, 'B': True},  # Run jump right
            5: {'left': True},  # Move left
            6: {'left': True, 'A': True},  # Run left
            7: {'left': True, 'B': True},  # Jump left
            8: {'left': True, 'A': True, 'B': True},  # Run jump left
            9: {'B': True},  # Jump
            10: {'A': True},  # Run/Fire
            11: {'down': True}  # Duck
        }
        return action_map.get(action_id, {})
    
    async def _handle_init(self, data: dict):
        """Handle initialization message."""
        logger.info(f"Received init message: {data}")
        
        # Send acknowledgment
        response = {
            'type': 'init_ack',
            'timestamp': int(time.time() * 1000),
            'status': 'ready',
            'training_mode': 'test_mode',
            'frame_skip': 1
        }
        
        await self.websocket_server._send_json(response)
        logger.info("Sent init acknowledgment")
    
    async def _handle_ping(self, data: dict):
        """Handle ping message."""
        timestamp = data.get('timestamp', int(time.time() * 1000))
        current_time = int(time.time() * 1000)
        latency = current_time - timestamp
        
        logger.info(f"Received ping, latency: {latency}ms")
        
        response = {
            'type': 'pong',
            'timestamp': current_time,
            'latency_ms': latency
        }
        
        await self.websocket_server._send_json(response)
    
    async def _handle_episode_event(self, data: dict):
        """Handle episode event."""
        event = data.get('event')
        episode_id = data.get('episode_id')
        logger.info(f"Episode {episode_id} event: {event}")
    
    async def _handle_frame_advance(self, data: dict):
        """Handle frame advance notification."""
        frame_id = data.get('frame_id')
        logger.debug(f"Frame advance: {frame_id}")
    
    async def _handle_error(self, data: dict):
        """Handle error message."""
        error_code = data.get('error_code')
        message = data.get('message')
        logger.error(f"Lua script error [{error_code}]: {message}")
    
    async def start_test(self, duration: int = 60):
        """Start communication test."""
        logger.info(f"Starting communication test on {self.host}:{self.port}")
        logger.info(f"Test will run for {duration} seconds")
        logger.info("Make sure FCEUX is running with the Lua script loaded!")
        
        try:
            # Start WebSocket server
            await self.websocket_server.start_server()
            logger.info("WebSocket server started, waiting for connections...")
            
            self.test_running = True
            start_time = time.time()
            
            # Wait for connection
            connection_timeout = 30  # 30 seconds
            while not self.websocket_server.is_client_connected() and time.time() - start_time < connection_timeout:
                await asyncio.sleep(1)
                elapsed = int(time.time() - start_time)
                if elapsed % 5 == 0:  # Log every 5 seconds
                    logger.info(f"Waiting for Lua script connection... ({elapsed}s elapsed)")
            
            if not self.websocket_server.is_client_connected():
                logger.error("No connection received within timeout period")
                logger.error("Check that:")
                logger.error("1. FCEUX is running")
                logger.error("2. The Lua script is loaded")
                logger.error("3. LuaSocket is working in FCEUX")
                return
            
            logger.info("Lua script connected! Starting communication test...")
            
            # Run test for specified duration
            test_start = time.time()
            last_stats_time = test_start
            
            while self.test_running and time.time() - test_start < duration:
                await asyncio.sleep(1)
                
                # Print stats every 10 seconds
                current_time = time.time()
                if current_time - last_stats_time >= 10:
                    elapsed = int(current_time - test_start)
                    logger.info(f"Test progress: {elapsed}s elapsed, "
                              f"Game states received: {self.game_state_count}, "
                              f"Actions sent: {self.action_count}")
                    last_stats_time = current_time
                
                # Check if connection is still active
                if not self.websocket_server.is_client_connected():
                    logger.warning("Connection lost during test")
                    break
            
            # Final stats
            total_time = time.time() - test_start
            logger.info("=== Test Results ===")
            logger.info(f"Test duration: {total_time:.1f} seconds")
            logger.info(f"Game states received: {self.game_state_count}")
            logger.info(f"Actions sent: {self.action_count}")
            
            if self.game_state_count > 0:
                logger.info(f"Average rate: {self.game_state_count / total_time:.1f} game states/second")
                logger.info("✅ Communication test PASSED - data is flowing!")
            else:
                logger.error("❌ Communication test FAILED - no game state data received")
                logger.error("This indicates the Lua script is not sending data")
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            raise
        finally:
            self.test_running = False
            await self.websocket_server.stop_server()
            logger.info("Communication test completed")
    
    async def stop_test(self):
        """Stop the test."""
        logger.info("Stopping communication test...")
        self.test_running = False


async def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test WebSocket communication with Lua script")
    parser.add_argument("--host", default="localhost", help="WebSocket host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Run test
    tester = CommunicationTester(args.host, args.port)
    
    try:
        await tester.start_test(args.duration)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        await tester.stop_test()


if __name__ == "__main__":
    asyncio.run(main())