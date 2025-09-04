import cv2
import numpy as np
import time
from typing import Optional, Tuple, Generator
from collections import deque
import logging

from app.utils.data_types import Frame

logger = logging.getLogger(__name__)


class VideoCapture:
    def __init__(self, 
                 device_index: int = 0,
                 target_size: Tuple[int, int] = (1280, 720),
                 target_fps: int = 30,
                 mirror: bool = True,
                 min_fps: float = 15,
                 downscale_steps: list = None):
        
        self.device_index = device_index
        self.target_size = target_size
        self.target_fps = target_fps
        self.mirror = mirror
        self.min_fps = min_fps
        self.downscale_steps = downscale_steps or [0.75, 0.5625]
        
        self.current_scale = 1.0
        self.current_size = target_size
        
        self.cap = None
        self.frame_counter = 0
        self.fps_history = deque(maxlen=60)
        self.last_frame_time = 0
        
        self._init_camera()
    
    def _init_camera(self):
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.device_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera device {self.device_index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.current_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.current_size[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
    
    def _update_fps(self) -> float:
        current_time = time.time()
        if self.last_frame_time > 0:
            fps = 1.0 / (current_time - self.last_frame_time)
            self.fps_history.append(fps)
        self.last_frame_time = current_time
        
        if len(self.fps_history) > 30:
            return np.mean(self.fps_history)
        return self.target_fps
    
    def _check_downscale(self, avg_fps: float):
        if avg_fps < self.min_fps and self.current_scale > min(self.downscale_steps):
            next_scale_idx = 0
            for i, scale in enumerate(self.downscale_steps):
                if scale < self.current_scale:
                    next_scale_idx = i
                    break
            
            if next_scale_idx < len(self.downscale_steps):
                self.current_scale = self.downscale_steps[next_scale_idx]
                new_width = int(self.target_size[0] * self.current_scale)
                new_height = int(self.target_size[1] * self.current_scale)
                self.current_size = (new_width, new_height)
                
                logger.warning(f"Downscaling to {new_width}x{new_height} (scale: {self.current_scale})")
                self._init_camera()
    
    def read_frame(self) -> Optional[Frame]:
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            return None
        
        if self.mirror:
            frame = cv2.flip(frame, 1)
        
        if frame.shape[:2] != (self.current_size[1], self.current_size[0]):
            frame = cv2.resize(frame, self.current_size)
        
        avg_fps = self._update_fps()
        if self.frame_counter % 90 == 0:  # Check every 3 seconds at 30fps
            self._check_downscale(avg_fps)
        
        ts_ms = int(time.time() * 1000)
        
        frame_obj = Frame(
            frame_id=self.frame_counter,
            ts_ms=ts_ms,
            image_bgr=frame,
            scale=self.current_scale
        )
        
        self.frame_counter += 1
        return frame_obj
    
    def stream_frames(self) -> Generator[Frame, None, None]:
        while True:
            frame = self.read_frame()
            if frame is not None:
                yield frame
    
    def get_current_fps(self) -> float:
        if len(self.fps_history) > 0:
            return np.mean(self.fps_history)
        return 0.0
    
    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __del__(self):
        self.release()


class VideoDisplay:
    def __init__(self, window_name: str = "Real-time Emotion Detection"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.last_display_time = 0
        self.display_fps_limit = 30
    
    def show_frame(self, frame: np.ndarray, force: bool = False) -> bool:
        current_time = time.time()
        time_since_last = current_time - self.last_display_time
        min_interval = 1.0 / self.display_fps_limit
        
        if not force and time_since_last < min_interval:
            return True
        
        cv2.imshow(self.window_name, frame)
        self.last_display_time = current_time
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        elif key == ord('r'):
            return 'reset'
        elif key == ord('e'):
            return 'toggle_emotion'
        
        return True
    
    def destroy(self):
        cv2.destroyWindow(self.window_name)
    
    def __del__(self):
        self.destroy()