"""
Frame capture system for Super Mario Bros AI training.

Handles window capture from FCEUX using cv2, frame synchronization,
and preprocessing for neural network input.
"""

import cv2
import numpy as np
import time
import threading
import logging
from typing import Optional, Tuple, Callable, List, Dict, Any
from collections import deque
import win32gui
import win32ui
import win32con
import win32api


class WindowCapture:
    """Handles window capture from FCEUX emulator."""
    
    def __init__(self, window_title: str = "FCEUX"):
        """
        Initialize window capture.
        
        Args:
            window_title: Title of the window to capture
        """
        self.window_title = window_title
        self.hwnd = None
        self.window_rect = None
        self.capture_region = None  # (x, y, width, height) for game area
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Find window
        self._find_window()
    
    def _find_window(self) -> bool:
        """
        Find FCEUX window.
        
        Returns:
            True if window found, False otherwise
        """
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if self.window_title.lower() in window_text.lower():
                    windows.append((hwnd, window_text))
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        if windows:
            self.hwnd, window_text = windows[0]
            self.window_rect = win32gui.GetWindowRect(self.hwnd)
            self.logger.info(f"Found window: {window_text} at {self.window_rect}")
            
            # Calculate game area (assuming standard FCEUX layout)
            # FCEUX typically has menu bars and borders, game area is usually centered
            self._calculate_capture_region()
            return True
        else:
            self.logger.warning(f"Window '{self.window_title}' not found")
            return False
    
    def _calculate_capture_region(self):
        """Calculate the game area within the FCEUX window."""
        if not self.window_rect:
            return
        
        # FCEUX window dimensions
        window_width = self.window_rect[2] - self.window_rect[0]
        window_height = self.window_rect[3] - self.window_rect[1]
        
        # Estimate game area (NES resolution is 256x240)
        # Account for window decorations and menu bars
        title_bar_height = 30
        menu_bar_height = 25
        border_width = 8
        
        # Calculate scaling factor
        available_width = window_width - (2 * border_width)
        available_height = window_height - title_bar_height - menu_bar_height - border_width
        
        # NES aspect ratio is 256:240 (approximately 1.067:1)
        nes_aspect = 256.0 / 240.0
        
        if available_width / available_height > nes_aspect:
            # Height constrained
            game_height = available_height
            game_width = int(game_height * nes_aspect)
        else:
            # Width constrained
            game_width = available_width
            game_height = int(game_width / nes_aspect)
        
        # Center the game area
        game_x = border_width + (available_width - game_width) // 2
        game_y = title_bar_height + menu_bar_height + (available_height - game_height) // 2
        
        self.capture_region = (game_x, game_y, game_width, game_height)
        self.logger.info(f"Game capture region: {self.capture_region}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture frame from FCEUX window.
        
        Returns:
            Captured frame as numpy array or None if capture failed
        """
        if not self.hwnd or not self.capture_region:
            if not self._find_window():
                return None
        
        try:
            # Get window device context
            hwndDC = win32gui.GetWindowDC(self.hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # Get capture region
            x, y, width, height = self.capture_region
            
            # Create bitmap
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            
            # Copy window content to bitmap
            saveDC.BitBlt((0, 0), (width, height), mfcDC, (x, y), win32con.SRCCOPY)
            
            # Convert to numpy array
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            
            frame = np.frombuffer(bmpstr, dtype='uint8')
            frame.shape = (height, width, 4)  # BGRA format
            
            # Convert BGRA to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            
            # Cleanup
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwndDC)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Frame capture failed: {e}")
            return None
    
    def is_window_available(self) -> bool:
        """Check if FCEUX window is available."""
        if not self.hwnd:
            return self._find_window()
        
        try:
            return win32gui.IsWindow(self.hwnd) and win32gui.IsWindowVisible(self.hwnd)
        except:
            return False


class GameFramePreprocessor:
    """Handles frame preprocessing for neural network input.
    
    NOTE: This is distinct from utils.preprocessing.FramePreprocessor which
    handles tensor conversions. This class works with raw numpy frames from
    window capture.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (84, 84)):
        """
        Initialize frame preprocessor.
        
        Args:
            target_size: Target frame size for neural network
        """
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for neural network input.
        
        Args:
            frame: Raw captured frame
            
        Returns:
            Preprocessed frame
        """
        if frame is None:
            return np.zeros((*self.target_size, 1), dtype=np.float32)
        
        try:
            # Convert to grayscale based on input format
            if len(frame.shape) == 3:
                # Multi-channel image
                if frame.shape[2] == 3:
                    # RGB image
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                elif frame.shape[2] == 4:
                    # RGBA image
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
                else:
                    # Unknown format, take first channel
                    gray_frame = frame[:, :, 0]
            elif len(frame.shape) == 2:
                # Already grayscale
                gray_frame = frame
            else:
                # Invalid shape, create empty frame
                self.logger.warning(f"Invalid frame shape: {frame.shape}")
                return np.zeros((*self.target_size, 1), dtype=np.float32)
            
            # Ensure grayscale frame is 2D
            if len(gray_frame.shape) > 2:
                gray_frame = gray_frame[:, :, 0]
            
            # Resize to target size
            resized_frame = cv2.resize(gray_frame, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Normalize pixel values to [0, 1]
            normalized_frame = resized_frame.astype(np.float32) / 255.0
            
            # Add channel dimension
            processed_frame = np.expand_dims(normalized_frame, axis=-1)
            
            return processed_frame
            
        except Exception as e:
            self.logger.error(f"Frame preprocessing failed: {e}")
            self.logger.error(f"Frame shape: {frame.shape if frame is not None else 'None'}")
            self.logger.error(f"Frame dtype: {frame.dtype if frame is not None else 'None'}")
            return np.zeros((*self.target_size, 1), dtype=np.float32)
    
    def preprocess_frame_stack(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess a stack of frames.
        
        Args:
            frames: List of frames to stack
            
        Returns:
            Stacked and preprocessed frames
        """
        processed_frames = []
        
        for frame in frames:
            processed_frame = self.preprocess_frame(frame)
            processed_frames.append(processed_frame)
        
        # Stack along channel dimension
        if processed_frames:
            stacked_frames = np.concatenate(processed_frames, axis=-1)
        else:
            # Return empty stack
            stacked_frames = np.zeros((*self.target_size, len(frames)), dtype=np.float32)
        
        return stacked_frames


def decode_lua_screen_frame(data: bytes, target_size: Tuple[int, int] = (84, 84)) -> Optional[np.ndarray]:
    """
    Decode compact screen frame from Lua's downsampled gui.gdscreenshot().
    
    Lua sends the frame already downsampled to 84x84 grayscale:
      Byte 0:     width  (84)
      Byte 1:     height (84)
      Bytes 2+:   width*height raw grayscale bytes (row-major, 0-255)
    
    Total: 2 + 84*84 = 7058 bytes (vs ~245KB for raw GD format)
    
    Args:
        data: Compact frame bytes (header already stripped by WebSocket handler)
        target_size: Expected size (used for validation only)
        
    Returns:
        Grayscale numpy array (height, width) normalized to [0, 1], or None on error
    """
    try:
        if len(data) < 2:
            return None
        
        width = data[0]
        height = data[1]
        
        if width <= 0 or height <= 0 or width > 256 or height > 256:
            return None
        
        expected_size = 2 + (width * height)
        if len(data) < expected_size:
            return None
        
        # Extract raw grayscale pixels
        pixel_data = data[2:2 + width * height]
        gray = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width)
        
        # Resize if dimensions don't match target (shouldn't normally happen)
        if (width, height) != target_size:
            gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        normalized = gray.astype(np.float32) / 255.0
        
        return normalized
        
    except Exception:
        return None


class FrameCapture:
    """
    Main frame capture system that coordinates window capture,
    preprocessing, and synchronization.
    
    Supports two capture modes:
    1. Lua screen capture (preferred): receives frames from gui.gdscreenshot() via WebSocket
    2. Win32 GDI capture (fallback): captures FCEUX window pixels directly
    """
    
    def __init__(self,
                 window_title: str = "FCEUX",
                 target_fps: int = 30,
                 frame_stack_size: int = 4,
                 target_size: Tuple[int, int] = (84, 84)):
        """
        Initialize frame capture system.
        
        Args:
            window_title: FCEUX window title
            target_fps: Target capture frame rate
            frame_stack_size: Number of frames to stack
            target_size: Target frame size for neural network
        """
        self.preprocessor = GameFramePreprocessor(target_size)
        self.target_size = target_size
        
        # Lua capture mode (preferred -- no window visibility needed)
        self._lua_capture_enabled = False
        self._lua_frame_count = 0
        
        # Win32 capture mode (fallback)
        try:
            self.window_capture = WindowCapture(window_title)
            self._use_mss = False
        except Exception:
            try:
                from mss import mss
                self._mss = mss()
                self._use_mss = True
            except ImportError:
                self._use_mss = False
        
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.frame_stack_size = frame_stack_size
        
        # Frame buffers
        self.raw_frame_buffer = deque(maxlen=frame_stack_size)
        self.processed_frame_buffer = deque(maxlen=frame_stack_size)
        
        # Capture state
        self.is_capturing = False
        self.capture_thread = None
        self.last_capture_time = 0
        
        # Frame callbacks
        self.frame_callbacks: List[Callable] = []
        
        # Statistics
        self.capture_stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'avg_capture_time': 0.0,
            'avg_fps': 0.0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def handle_lua_screen_frame(self, gd_data: bytes):
        """
        Handle screen frame received from Lua's gui.gdscreenshot().
        
        This is the preferred capture mode -- frames come directly from the
        emulator internals, no window visibility required.
        
        Args:
            gd_data: Raw GD format image bytes
        """
        frame = decode_lua_screen_frame(gd_data, self.target_size)
        if frame is not None:
            # Add channel dimension: (84, 84) -> (84, 84, 1)
            processed = np.expand_dims(frame, axis=-1)
            self.processed_frame_buffer.append(processed)
            self._lua_capture_enabled = True
            self._lua_frame_count += 1
            self.capture_stats['frames_captured'] += 1
            
            # Notify callbacks
            for callback in self.frame_callbacks:
                try:
                    callback(processed, time.time())
                except Exception as e:
                    self.logger.error(f"Frame callback error: {e}")
        else:
            self.capture_stats['frames_dropped'] += 1
    
    def register_frame_callback(self, callback: Callable[[np.ndarray, float], None]):
        """
        Register callback for captured frames.
        
        Args:
            callback: Function to call with (frame, timestamp)
        """
        self.frame_callbacks.append(callback)
    
    def start_capture(self):
        """Start frame capture in background thread.
        
        If Lua screen capture is active (frames arriving via handle_lua_screen_frame),
        the Win32 capture thread is not started -- Lua frames are preferred.
        """
        if self.is_capturing:
            self.logger.warning("Frame capture already running")
            return
        
        if self._lua_capture_enabled:
            self.logger.info("Using Lua screen capture mode (no window capture needed)")
            self.is_capturing = True
            return
        
        # Fallback: try Win32 GDI window capture
        if not hasattr(self, 'window_capture') or not self.window_capture.is_window_available():
            self.logger.warning("FCEUX window not found for GDI capture")
            self.logger.warning("Frames will come from Lua gui.gdscreenshot() when available")
            self.is_capturing = True  # Mark as running even without GDI
            return
        
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.logger.info(f"Started Win32 frame capture at {self.target_fps} FPS")
    
    def stop_capture(self):
        """Stop frame capture."""
        if not self.is_capturing:
            return
        
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        
        self.logger.info("Stopped frame capture")
    
    def _capture_loop(self):
        """Main capture loop running in background thread."""
        frame_times = deque(maxlen=60)  # Track last 60 frame times
        
        while self.is_capturing:
            start_time = time.time()
            
            # Rate limiting
            time_since_last = start_time - self.last_capture_time
            if time_since_last < self.frame_interval:
                sleep_time = self.frame_interval - time_since_last
                time.sleep(sleep_time)
                start_time = time.time()
            
            # Capture frame
            raw_frame = self.window_capture.capture_frame()
            capture_time = time.time()
            
            if raw_frame is not None:
                # Preprocess frame
                processed_frame = self.preprocessor.preprocess_frame(raw_frame)
                
                # Update buffers
                self.raw_frame_buffer.append(raw_frame)
                self.processed_frame_buffer.append(processed_frame)
                
                # Update statistics
                self.capture_stats['frames_captured'] += 1
                frame_time = capture_time - start_time
                self.capture_stats['avg_capture_time'] = (
                    (self.capture_stats['avg_capture_time'] * 0.9) + (frame_time * 0.1)
                )
                
                frame_times.append(capture_time)
                if len(frame_times) > 1:
                    fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                    self.capture_stats['avg_fps'] = fps
                
                # Notify callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(processed_frame, capture_time)
                    except Exception as e:
                        self.logger.error(f"Error in frame callback: {e}")
                
            else:
                self.capture_stats['frames_dropped'] += 1
                
                # Try to reconnect to window
                if not self.window_capture.is_window_available():
                    self.logger.warning("FCEUX window lost, attempting to reconnect...")
                    time.sleep(1.0)  # Wait before retry
            
            self.last_capture_time = capture_time
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest processed frame."""
        if self.processed_frame_buffer:
            return self.processed_frame_buffer[-1]
        return None
    
    def get_frame_stack(self) -> np.ndarray:
        """
        Get current frame stack for neural network input.
        
        Returns:
            Stacked frames with shape (height, width, stack_size)
        """
        try:
            frames = list(self.processed_frame_buffer)
            
            # Pad with zeros if not enough frames
            while len(frames) < self.frame_stack_size:
                frames.insert(0, np.zeros((*self.preprocessor.target_size, 1), dtype=np.float32))
            
            # Take only the most recent frames
            frames = frames[-self.frame_stack_size:]
            
            # Validate frames before stacking
            valid_frames = []
            for frame in frames:
                if isinstance(frame, np.ndarray) and frame.size > 0:
                    valid_frames.append(frame)
                else:
                    # Add default frame if invalid
                    valid_frames.append(np.zeros((*self.preprocessor.target_size, 1), dtype=np.float32))
            
            # Stack frames
            if valid_frames:
                return self.preprocessor.preprocess_frame_stack(valid_frames)
            else:
                # Return default frame stack
                return np.zeros((*self.preprocessor.target_size, self.frame_stack_size), dtype=np.float32)
                
        except Exception as e:
            self.logger.error(f"Error in get_frame_stack: {e}")
            # Return default frame stack on error
            return np.zeros((*self.preprocessor.target_size, self.frame_stack_size), dtype=np.float32)
    
    def get_raw_frame_stack(self) -> List[np.ndarray]:
        """Get raw frame stack for debugging."""
        return list(self.raw_frame_buffer)
    
    def is_window_available(self) -> bool:
        """Check if FCEUX window is available."""
        return self.window_capture.is_window_available()
    
    def get_capture_stats(self) -> dict:
        """Get capture statistics."""
        stats = self.capture_stats.copy()
        stats.update({
            'is_capturing': self.is_capturing,
            'window_available': self.is_window_available(),
            'buffer_size': len(self.processed_frame_buffer),
            'target_fps': self.target_fps,
            'frame_stack_size': self.frame_stack_size
        })
        
        return stats
    
    def parse_game_state(self, game_state_data: bytes) -> Dict[str, Any]:
        """
        Parse binary game state payload from Lua script.
        
        IMPORTANT: The WebSocket server already strips the 8-byte header before
        calling the binary handler.  This method receives the RAW PAYLOAD only
        (expected to be exactly 128 bytes).  Do NOT try to parse a header here.
        
        Binary payload layout (128 bytes total):
          Bytes  0-15  : Mario Data Block  (16 bytes)
          Bytes 16-47  : Enemy Data Block   (32 bytes)
          Bytes 48-111 : Level Data Block   (64 bytes)
          Bytes 112-127: Game Variables     (16 bytes)
        
        Args:
            game_state_data: 128-byte binary payload (header already stripped)
            
        Returns:
            Parsed game state dictionary
        """
        import struct
        
        MARIO_FMT  = struct.Struct('<HHbbBBBBBBBB')  # 16 bytes (includes 2 reserved)
        ENEMY_COUNT = 8
        ENEMY_BYTES_PER = 4
        ENEMY_TOTAL = ENEMY_COUNT * ENEMY_BYTES_PER  # 32 bytes
        LEVEL_BLOCK_SIZE = 64
        GAMEVARS_BLOCK_SIZE = 16
        EXPECTED_PAYLOAD = 128
        
        def _hexdump(data: bytes, limit: int = 64) -> str:
            return ' '.join(f'{b:02x}' for b in data[:limit])
        
        try:
            total_len = len(game_state_data)
            self.logger.debug(f"[GS] payload={total_len} (expected {EXPECTED_PAYLOAD}). Hex head: {_hexdump(game_state_data, 32)}")

            if total_len < MARIO_FMT.size:
                self.logger.warning(f"[GS] Payload too short for Mario block: {total_len} bytes")
                return self._get_default_game_state()

            # ----- Mario (16B) at offset 0 -----
            off = 0
            mario_end = off + MARIO_FMT.size
            try:
                mario_values = MARIO_FMT.unpack(game_state_data[off:mario_end])
            except struct.error as e:
                self.logger.error(f"[GS] Mario unpack failed: {e}")
                return self._get_default_game_state()
            off = mario_end

            # ----- Enemy (32B) at offset 16 -----
            enemy_end = off + ENEMY_TOTAL
            if total_len >= enemy_end:
                off = enemy_end
            else:
                self.logger.debug(f"[GS] Enemy data short; proceeding with zeros")
                off = total_len  # skip what we can

            # ----- Level Data (64B) at offset 48 -----
            tail_len = total_len - off
            level_available = min(tail_len, LEVEL_BLOCK_SIZE)
            level_block = game_state_data[off: off + level_available]
            off += level_available

            # ----- Game Variables (16B) at offset 112 -----
            gv_available = min(total_len - off, GAMEVARS_BLOCK_SIZE)
            gamevars_block = game_state_data[off: off + gv_available]

            self.logger.debug(
                f"[GS] Segments: mario=16, enemy=32, "
                f"level=({len(level_block)}/{LEVEL_BLOCK_SIZE}), gamevars=({len(gamevars_block)}/{GAMEVARS_BLOCK_SIZE})"
            )

            # ---- Interpret Mario values ----
            # <HHbbBBBBBBBB
            # x_pos_world(u16), y_pos_level(u16), x_vel(i8), y_vel(i8),
            # power(u8), anim(u8), dir(u8), player_state(u8), lives(u8), invinc(u8), rsv1, rsv2
            mx       = mario_values[0]
            my       = mario_values[1]
            mx_vel   = mario_values[2]
            my_vel   = mario_values[3]
            mpower   = mario_values[4]
            mstate   = mario_values[7]
            mlives   = mario_values[8]

            # ---- Interpret Level block ----
            world = 1
            level = 1
            score = 0
            coins = 0
            time_rem = 400

            if len(level_block) >= 14:
                try:
                    camera_x = struct.unpack_from('<H', level_block, 0)[0]
                    world     = level_block[2]
                    level     = level_block[3]
                    s100k     = level_block[4]
                    s10k      = level_block[5]
                    s1k       = level_block[6]
                    s100      = level_block[7]
                    time_rem  = struct.unpack_from('<I', level_block, 8)[0]
                    coins     = struct.unpack_from('<H', level_block, 12)[0]
                    score     = (s100k * 100000) + (s10k * 10000) + (s1k * 1000) + (s100 * 100)
                except Exception as e:
                    self.logger.warning(f"[GS] Level parse failed ({len(level_block)}B): {e}; using defaults")

            self.logger.debug(f"PARSED: mario_x={int(mx)}, mario_y={int(my)}, lives={int(mlives)}, score={int(score)}")
            
            return {
                'mario_x': int(mx),
                'mario_y': int(my),
                'mario_x_vel': int(mx_vel),
                'mario_y_vel': int(my_vel),
                'mario_state': int(mstate),
                'world': int(world),
                'level': int(level),
                'lives': int(mlives),
                'score': int(score),
                'coins': int(coins),
                'time': int(time_rem),
                'powerup': int(mpower),
                'power_state': int(mpower),
                'timestamp': time.time(),
                '_payload_len': total_len,
            }

        except Exception as e:
            self.logger.error(f"[GS] Fatal parse error: {e}")
            self.logger.error(f"[GS] Payload len: {len(game_state_data)}; Hex head: {_hexdump(game_state_data, 32)}")
            return self._get_default_game_state()
    
    def _get_default_game_state(self) -> Dict[str, Any]:
        """Get default game state when parsing fails."""
        return {
            'mario_x': 0,
            'mario_y': 0,
            'mario_state': 0,
            'world': 1,
            'level': 1,
            'lives': 3,
            'score': 0,
            'coins': 0,
            'time': 400,
            'powerup': 0,
            'timestamp': time.time()
        }
    
    def process_frame(self, game_state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process game state and return frames and state vector for neural network.
        
        Args:
            game_state: Parsed game state dictionary
            
        Returns:
            Tuple of (processed_frames, state_vector)
        """
        try:
            # Get current frame stack
            frames = self.get_frame_stack()
            
            # Ensure frames have correct shape: (height, width, channels)
            if len(frames.shape) == 4 and frames.shape[0] == 1:
                # Remove batch dimension if present
                frames = frames.squeeze(0)
            elif len(frames.shape) == 2:
                # Add channel dimension if missing
                frames = np.expand_dims(frames, axis=-1)
            
            # Ensure we have the expected shape
            expected_shape = (*self.preprocessor.target_size, self.frame_stack_size)
            if frames.shape != expected_shape:
                self.logger.warning(f"Frame shape mismatch: got {frames.shape}, expected {expected_shape}")
                frames = np.zeros(expected_shape, dtype=np.float32)
            
            # Create state vector from game state (12 elements to match DQN model)
            # All features normalized to approximately [0, 1] or [-1, 1] range
            state_vector = np.array([
                game_state.get('mario_x', 0) / 3168.0,             # X position (World 1-1 length)
                game_state.get('mario_y', 0) / 240.0,              # Y position (screen height)
                game_state.get('mario_x_vel', 0) / 40.0,           # X velocity (signed, typical range -40 to 40)
                game_state.get('mario_y_vel', 0) / 40.0,           # Y velocity (signed, typical range -40 to 40)
                game_state.get('lives', 3) / 5.0,                  # Lives (max ~5)
                game_state.get('powerup', 0) / 2.0,                # Power state (0=small, 1=big, 2=fire)
                game_state.get('time', 400) / 400.0,               # Timer
                game_state.get('coins', 0) / 99.0,                 # Coins (max 99)
                game_state.get('score', 0) / 100000.0,             # Score
                float(game_state.get('world', 1)) / 8.0,           # World number
                float(game_state.get('level', 1)) / 4.0,           # Level number
                float(game_state.get('mario_state', 0)) / 11.0,    # Player state byte
            ], dtype=np.float32)
            
            # Ensure state vector has correct shape
            if len(state_vector.shape) != 1:
                state_vector = state_vector.flatten()
            
            return frames, state_vector
            
        except Exception as e:
            self.logger.error(f"Failed to process frame: {e}")
            self.logger.error(f"Game state keys: {list(game_state.keys()) if game_state else 'None'}")
            # Return default values
            default_frames = np.zeros((*self.preprocessor.target_size, self.frame_stack_size), dtype=np.float32)
            default_state = np.zeros(12, dtype=np.float32)
            return default_frames, default_state
    
    def reset_stats(self):
        """Reset capture statistics."""
        self.capture_stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'avg_capture_time': 0.0,
            'avg_fps': 0.0
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start_capture()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_capture()


# Utility functions

def test_frame_capture(duration: float = 10.0):
    """
    Test frame capture system.
    
    Args:
        duration: Test duration in seconds
    """
    logging.basicConfig(level=logging.INFO)
    
    def frame_callback(frame, timestamp):
        print(f"Captured frame at {timestamp:.3f}, shape: {frame.shape}")
    
    with FrameCapture() as capture:
        capture.register_frame_callback(frame_callback)
        
        print(f"Testing frame capture for {duration} seconds...")
        time.sleep(duration)
        
        stats = capture.get_capture_stats()
        print(f"Capture statistics: {stats}")


if __name__ == "__main__":
    test_frame_capture()