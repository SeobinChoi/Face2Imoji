import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from app.utils.data_types import Frame, PersonTrack

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    balloon_diameter_px: int = 64
    balloon_alpha: float = 0.7
    balloon_offset_y_px: int = -60
    balloon_font_scale: float = 0.6
    balloon_prob_digits: int = 2
    
    skeleton_line_thickness: int = 2
    skeleton_point_radius: int = 3
    
    hud_pos: str = "bottom_left"
    hud_alpha: float = 0.6
    hud_font_scale: float = 0.7
    
    logo_path: str = "assets/logo.png"
    logo_alpha: float = 0.5
    logo_pos: str = "top_right"


class Renderer:
    def __init__(self, config: Optional[VisualizationConfig] = None, emoji_mapping: Optional[Dict[str, str]] = None):
        self.config = config or VisualizationConfig()
        
        self.emoji_mapping = emoji_mapping or {
            "happy": "😄",
            "neutral": "😐",
            "surprise": "😲",
            "sad": "😢",
            "angry": "😡",
            "fear": "😨",
            "disgust": "🤢"
        }
        
        self.skeleton_connections = [
            (0, 1), (0, 2),  # Nose to eyes
            (1, 3), (2, 4),  # Eyes
            (3, 5), (4, 6),  # Eye to ear
            (5, 7), (6, 8),  # Ears
            (9, 10),  # Mouth
            (11, 12),  # Shoulders
            (11, 13), (12, 14),  # Shoulder to elbow
            (13, 15), (14, 16),  # Elbow to wrist
            (11, 23), (12, 24),  # Shoulder to hip
            (23, 24),  # Hips
            (23, 25), (24, 26),  # Hip to knee
            (25, 27), (26, 28)  # Knee to ankle
        ]
        
        self.logo = None
        self._load_logo()
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        logger.info("Renderer initialized with config")
    
    def _load_logo(self):
        try:
            logo = cv2.imread(self.config.logo_path, cv2.IMREAD_UNCHANGED)
            if logo is not None:
                self.logo = cv2.resize(logo, (100, 100))
                logger.info(f"Logo loaded from {self.config.logo_path}")
        except Exception as e:
            logger.warning(f"Could not load logo: {e}")
    
    def render(self, frame: Frame, tracks: List[PersonTrack], max_render: int = 6) -> np.ndarray:
        image = frame.image_bgr.copy()
        
        tracks_to_render = sorted(tracks, key=lambda t: t.id)[:max_render]
        
        for track in tracks_to_render:
            self._draw_skeleton(image, track)
        
        for track in tracks_to_render:
            self._draw_emotion_balloon(image, track)
        
        self._draw_hud(image, tracks)
        
        self._draw_logo(image)
        
        self._draw_demo_banner(image)
        
        return image
    
    def _draw_skeleton(self, image: np.ndarray, track: PersonTrack):
        color = track.get_display_color()
        keypoints = track.keypoints
        
        if not keypoints:
            return
        
        for connection in self.skeleton_connections:
            pt1_idx, pt2_idx = connection
            
            if pt1_idx >= len(keypoints) or pt2_idx >= len(keypoints):
                continue
            
            pt1 = keypoints[pt1_idx]
            pt2 = keypoints[pt2_idx]
            
            if pt1.conf > 0.3 and pt2.conf > 0.3:
                x1, y1 = int(pt1.x), int(pt1.y)
                x2, y2 = int(pt2.x), int(pt2.y)
                
                cv2.line(image, (x1, y1), (x2, y2), color, self.config.skeleton_line_thickness)
        
        for kp in keypoints:
            if kp.conf > 0.3:
                x, y = int(kp.x), int(kp.y)
                cv2.circle(image, (x, y), self.config.skeleton_point_radius, color, -1)
    
    def _draw_emotion_balloon(self, image: np.ndarray, track: PersonTrack):
        if not track.emotion_label:
            return
        
        emoji = self.emoji_mapping.get(track.emotion_label, "😐")
        
        cx, cy = track.body_center
        if cx == 0 and cy == 0:
            return
        
        cx, cy = int(cx), int(cy)
        
        shoulder_indices = [11, 12]
        shoulder_y = cy
        if len(track.keypoints) > max(shoulder_indices):
            shoulder_ys = []
            for idx in shoulder_indices:
                if track.keypoints[idx].conf > 0.3:
                    shoulder_ys.append(track.keypoints[idx].y)
            if shoulder_ys:
                shoulder_y = int(min(shoulder_ys))
        
        balloon_y = shoulder_y + self.config.balloon_offset_y_px
        balloon_x = cx
        
        balloon_y = max(40, min(balloon_y, image.shape[0] - 40))
        balloon_x = max(40, min(balloon_x, image.shape[1] - 40))
        
        overlay = image.copy()
        
        radius = self.config.balloon_diameter_px // 2
        cv2.circle(overlay, (balloon_x, balloon_y), radius, (255, 255, 255), -1)
        cv2.circle(overlay, (balloon_x, balloon_y), radius, (200, 200, 200), 2)
        
        cv2.addWeighted(overlay, self.config.balloon_alpha, image, 1 - self.config.balloon_alpha, 0, image)
        
        emoji_size = int(radius * 1.2)
        self._draw_emoji_text(image, emoji, (balloon_x, balloon_y - 5), emoji_size)
        
        if track.emotion_prob and track.emotion_label in track.emotion_prob:
            prob = track.emotion_prob[track.emotion_label] * 100
            prob_text = f"{prob:.{self.config.balloon_prob_digits}f}%"
            
            text_size = cv2.getTextSize(prob_text, self.font, self.config.balloon_font_scale * 0.6, 1)[0]
            text_x = balloon_x - text_size[0] // 2
            text_y = balloon_y + radius - 10
            
            cv2.putText(image, prob_text, (text_x, text_y),
                       self.font, self.config.balloon_font_scale * 0.6,
                       (100, 100, 100), 1, cv2.LINE_AA)
    
    def _draw_emoji_text(self, image: np.ndarray, emoji: str, center: Tuple[int, int], size: int):
        try:
            cv2.putText(image, emoji, 
                       (center[0] - size//2, center[1] + size//3),
                       cv2.FONT_HERSHEY_COMPLEX, size/40,
                       (0, 0, 0), 2, cv2.LINE_AA)
        except:
            fallback_text = {
                "😄": ":)",
                "😐": ":|",
                "😲": ":O",
                "😢": ":(",
                "😡": ">:(",
                "😨": ":S",
                "🤢": ":P"
            }.get(emoji, "?")
            
            cv2.putText(image, fallback_text,
                       (center[0] - 15, center[1] + 5),
                       self.font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    def _draw_hud(self, image: np.ndarray, tracks: List[PersonTrack]):
        h, w = image.shape[:2]
        
        confirmed_tracks = [t for t in tracks if t.emotion_label]
        num_people = len(confirmed_tracks)
        num_happy = sum(1 for t in confirmed_tracks if t.emotion_label == "happy")
        
        top_mood = "neutral"
        top_mood_prob = 0.0
        if confirmed_tracks:
            all_emotions = {}
            for label in self.emoji_mapping.keys():
                probs = [t.emotion_prob.get(label, 0.0) for t in confirmed_tracks if t.emotion_prob]
                if probs:
                    all_emotions[label] = np.mean(probs)
            
            if all_emotions:
                top_mood = max(all_emotions, key=all_emotions.get)
                top_mood_prob = all_emotions[top_mood] * 100
        
        lines = [
            f"People: {num_people}",
            f"Smiling: {num_happy}",
            f"Top mood: {self.emoji_mapping.get(top_mood, '?')} ({top_mood_prob:.0f}%)"
        ]
        
        if self.config.hud_pos == "bottom_left":
            x, y = 10, h - 80
        else:
            x, y = 10, 100
        
        box_h = len(lines) * 30 + 20
        box_w = 250
        
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (50, 50, 50), -1)
        cv2.addWeighted(overlay, self.config.hud_alpha, image, 1 - self.config.hud_alpha, 0, image)
        
        for i, line in enumerate(lines):
            cv2.putText(image, line,
                       (x + 10, y + 25 + i * 30),
                       self.font, self.config.hud_font_scale,
                       (255, 255, 255), 1, cv2.LINE_AA)
    
    def _draw_logo(self, image: np.ndarray):
        if self.logo is None:
            return
        
        h, w = image.shape[:2]
        logo_h, logo_w = self.logo.shape[:2]
        
        if self.config.logo_pos == "top_right":
            x = w - logo_w - 10
            y = 10
        else:
            x = 10
            y = 10
        
        if x + logo_w <= w and y + logo_h <= h:
            roi = image[y:y+logo_h, x:x+logo_w]
            
            if self.logo.shape[2] == 4:  # Has alpha channel
                alpha = self.logo[:, :, 3] / 255.0 * self.config.logo_alpha
                for c in range(3):
                    roi[:, :, c] = roi[:, :, c] * (1 - alpha) + self.logo[:, :, c] * alpha
            else:
                cv2.addWeighted(self.logo[:, :, :3], self.config.logo_alpha,
                              roi, 1 - self.config.logo_alpha, 0, roi)
    
    def _draw_demo_banner(self, image: np.ndarray):
        h, w = image.shape[:2]
        
        text = "Real-time Demo (Not Recording)"
        text_size = cv2.getTextSize(text, self.font, 0.5, 1)[0]
        
        x = (w - text_size[0]) // 2
        y = h - 10
        
        cv2.putText(image, text, (x, y),
                   self.font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)