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
from typing import Optional, Tuple, Callable, List
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


class FramePreprocessor:
    """Handles frame preprocessing for neural network input."""
    
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
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray_frame = frame
            
            # Resize to target size
            resized_frame = cv2.resize(gray_frame, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Normalize pixel values to [0, 1]
            normalized_frame = resized_frame.astype(np.float32) / 255.0
            
            # Add channel dimension
            processed_frame = np.expand_dims(normalized_frame, axis=-1)
            
            return processed_frame
            
        except Exception as e:
            self.logger.error(f"Frame preprocessing failed: {e}")
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


class FrameCapture:
    """
    Main frame capture system that coordinates window capture,
    preprocessing, and synchronization.
    """
    
    def __init__(self, 
                 window_title: str = "FCEUX",
                 target_fps: int = 60,
                 frame_stack_size: int = 8,
                 target_size: Tuple[int, int] = (84, 84)):
        """
        Initialize frame capture system.
        
        Args:
            window_title: FCEUX window title
            target_fps: Target capture frame rate
            frame_stack_size: Number of frames to stack
            target_size: Target frame size for neural network
        """
        self.window_capture = WindowCapture(window_title)
        self.preprocessor = FramePreprocessor(target_size)
        
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
    
    def register_frame_callback(self, callback: Callable[[np.ndarray, float], None]):
        """
        Register callback for captured frames.
        
        Args:
            callback: Function to call with (frame, timestamp)
        """
        self.frame_callbacks.append(callback)
    
    def start_capture(self):
        """Start frame capture in background thread."""
        if self.is_capturing:
            self.logger.warning("Frame capture already running")
            return
        
        if not self.window_capture.is_window_available():
            raise RuntimeError("FCEUX window not found")
        
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.logger.info(f"Started frame capture at {self.target_fps} FPS")
    
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
        frames = list(self.processed_frame_buffer)
        
        # Pad with zeros if not enough frames
        while len(frames) < self.frame_stack_size:
            frames.insert(0, np.zeros((*self.preprocessor.target_size, 1), dtype=np.float32))
        
        # Take only the most recent frames
        frames = frames[-self.frame_stack_size:]
        
        # Stack frames
        return self.preprocessor.preprocess_frame_stack(frames)
    
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