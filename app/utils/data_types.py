from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np
from enum import Enum


class TrackState(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


@dataclass
class Landmark:
    x: float
    y: float
    z: Optional[float] = None
    conf: float = 1.0
    
    def to_pixel(self, width: int, height: int) -> Tuple[int, int]:
        return (int(self.x * width), int(self.y * height))


@dataclass
class Frame:
    frame_id: int
    ts_ms: int
    image_bgr: np.ndarray
    image_rgb: Optional[np.ndarray] = None
    scale: float = 1.0
    
    def get_rgb(self) -> np.ndarray:
        if self.image_rgb is None:
            self.image_rgb = self.image_bgr[:, :, ::-1].copy()
        return self.image_rgb
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.image_bgr.shape
    
    @property
    def height(self) -> int:
        return self.image_bgr.shape[0]
    
    @property
    def width(self) -> int:
        return self.image_bgr.shape[1]


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x, y, w, h
    score: float
    
    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w/2, y + h/2)
    
    @property
    def area(self) -> float:
        return self.bbox[2] * self.bbox[3]
    
    def to_xywh_int(self) -> Tuple[int, int, int, int]:
        return tuple(int(v) for v in self.bbox)
    
    def to_xyxy(self) -> Tuple[float, float, float, float]:
        x, y, w, h = self.bbox
        return (x, y, x + w, y + h)


@dataclass
class PersonPose:
    keypoints: List[Landmark]
    score: float
    bbox_body: Optional[Tuple[float, float, float, float]] = None
    
    def compute_body_bbox(self) -> Tuple[float, float, float, float]:
        if not self.keypoints:
            return (0, 0, 0, 0)
        
        xs = [kp.x for kp in self.keypoints if kp.conf > 0.3]
        ys = [kp.y for kp in self.keypoints if kp.conf > 0.3]
        
        if not xs or not ys:
            return (0, 0, 0, 0)
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def get_body_center(self) -> Tuple[float, float]:
        shoulder_indices = [11, 12]
        hip_indices = [23, 24]
        
        points = []
        for idx in shoulder_indices + hip_indices:
            if idx < len(self.keypoints) and self.keypoints[idx].conf > 0.3:
                points.append((self.keypoints[idx].x, self.keypoints[idx].y))
        
        if not points:
            if self.bbox_body:
                x, y, w, h = self.bbox_body
                return (x + w/2, y + h/2)
            return (0.5, 0.5)
        
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        return (center_x, center_y)


@dataclass
class PersonTrack:
    id: int
    body_center: Tuple[float, float]
    bbox_body: Tuple[float, float, float, float]
    keypoints: List[Landmark]
    face_bbox: Optional[Tuple[float, float, float, float]] = None
    emotion_prob: Dict[str, float] = field(default_factory=dict)
    emotion_label: str = "neutral"
    last_update_ms: int = 0
    state: TrackState = TrackState.TENTATIVE
    age: int = 0
    hits: int = 0
    
    @property
    def color_id(self) -> int:
        return hash(self.id) % 360
    
    def get_display_color(self) -> Tuple[int, int, int]:
        import colorsys
        h = self.color_id / 360.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
        return (int(b * 255), int(g * 255), int(r * 255))


@dataclass
class EmotionResult:
    probabilities: Dict[str, float]
    label: str
    confidence: float