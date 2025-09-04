import time
import numpy as np
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    capture_ms: float = 0.0
    pose_ms: float = 0.0
    face_ms: float = 0.0
    tracking_ms: float = 0.0
    emotion_ms: float = 0.0
    viz_ms: float = 0.0
    total_ms: float = 0.0
    fps: float = 0.0


class PerformanceMonitor:
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.timings: Dict[str, deque] = {}
        self.start_times: Dict[str, float] = {}
        self.frame_times = deque(maxlen=window_size)
        self.last_frame_time = None
        
        self.stages = ["capture", "pose", "face", "tracking", "emotion", "viz"]
        for stage in self.stages:
            self.timings[stage] = deque(maxlen=window_size)
    
    def start_stage(self, stage: str):
        self.start_times[stage] = time.perf_counter() * 1000
    
    def end_stage(self, stage: str):
        if stage in self.start_times:
            elapsed = time.perf_counter() * 1000 - self.start_times[stage]
            self.timings[stage].append(elapsed)
            del self.start_times[stage]
            return elapsed
        return 0.0
    
    def frame_complete(self):
        current_time = time.perf_counter()
        if self.last_frame_time is not None:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
        self.last_frame_time = current_time
    
    def get_stats(self) -> TimingStats:
        stats = TimingStats()
        
        for stage in self.stages:
            if self.timings[stage]:
                avg_ms = np.mean(self.timings[stage])
                setattr(stats, f"{stage}_ms", avg_ms)
                stats.total_ms += avg_ms
        
        if self.frame_times:
            avg_frame_time = np.mean(self.frame_times)
            stats.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        return stats
    
    def get_current_fps(self) -> float:
        if len(self.frame_times) > 0:
            avg_frame_time = np.mean(self.frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        return 0.0
    
    def should_downscale(self, min_fps: float = 15) -> bool:
        current_fps = self.get_current_fps()
        if len(self.frame_times) >= 30:  # Need enough samples
            return current_fps < min_fps
        return False
    
    def log_stats(self):
        stats = self.get_stats()
        logger.info(f"Performance: FPS={stats.fps:.1f}, "
                   f"Total={stats.total_ms:.1f}ms "
                   f"(Cap={stats.capture_ms:.1f}, Pose={stats.pose_ms:.1f}, "
                   f"Face={stats.face_ms:.1f}, Track={stats.tracking_ms:.1f}, "
                   f"Emo={stats.emotion_ms:.1f}, Viz={stats.viz_ms:.1f})")


class AdaptiveProcessor:
    def __init__(self,
                 target_fps: float = 20,
                 min_fps: float = 15):
        
        self.target_fps = target_fps
        self.min_fps = min_fps
        
        self.process_every_n_frames = {
            'face': 1,
            'emotion': 3,
            'pose': 1
        }
        
        self.frame_counter = 0
        self.adjustment_cooldown = 0
    
    def should_process(self, component: str) -> bool:
        if component not in self.process_every_n_frames:
            return True
        
        n = self.process_every_n_frames[component]
        return self.frame_counter % n == 0
    
    def adapt(self, current_fps: float, queue_length: int = 0):
        self.frame_counter += 1
        
        if self.adjustment_cooldown > 0:
            self.adjustment_cooldown -= 1
            return
        
        if current_fps < self.min_fps:
            if self.process_every_n_frames['emotion'] < 5:
                self.process_every_n_frames['emotion'] += 1
                logger.info(f"Reducing emotion processing to every {self.process_every_n_frames['emotion']} frames")
            elif self.process_every_n_frames['face'] < 2:
                self.process_every_n_frames['face'] = 2
                logger.info("Reducing face detection to every 2 frames")
            
            self.adjustment_cooldown = 30
        
        elif current_fps > self.target_fps * 1.2:
            if self.process_every_n_frames['emotion'] > 3:
                self.process_every_n_frames['emotion'] -= 1
                logger.info(f"Increasing emotion processing to every {self.process_every_n_frames['emotion']} frames")
            elif self.process_every_n_frames['face'] > 1:
                self.process_every_n_frames['face'] = 1
                logger.info("Restoring face detection to every frame")
            
            self.adjustment_cooldown = 30
        
        if queue_length > 20:
            self.process_every_n_frames['emotion'] = min(self.process_every_n_frames['emotion'] + 1, 6)


class MemoryMonitor:
    def __init__(self):
        self.last_check_time = 0
        self.check_interval = 10.0  # seconds
    
    def should_check(self) -> bool:
        current_time = time.time()
        if current_time - self.last_check_time > self.check_interval:
            self.last_check_time = current_time
            return True
        return False
    
    def log_memory(self):
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory usage: {memory_mb:.1f} MB")
        except ImportError:
            pass