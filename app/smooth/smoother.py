import numpy as np
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class EmotionSmoother:
    def __init__(self,
                 ema_alpha: float = 0.4,
                 switch_delta: float = 0.15,
                 min_hold_frames: int = 5):
        
        self.ema_alpha = ema_alpha
        self.switch_delta = switch_delta
        self.min_hold_frames = min_hold_frames
        
        self.ema_states: Dict[int, Dict[str, float]] = defaultdict(dict)
        self.current_labels: Dict[int, str] = {}
        self.label_hold_counts: Dict[int, int] = defaultdict(int)
        self.pending_labels: Dict[int, str] = {}
        
        logger.info(f"EmotionSmoother initialized: ema_alpha={ema_alpha}, "
                   f"switch_delta={switch_delta}, min_hold_frames={min_hold_frames}")
    
    def smooth(self, 
               track_id: int,
               raw_probs: Dict[str, float]) -> Tuple[Dict[str, float], str]:
        
        if track_id not in self.ema_states:
            self.ema_states[track_id] = raw_probs.copy()
            self.current_labels[track_id] = max(raw_probs, key=raw_probs.get)
            self.label_hold_counts[track_id] = 0
            return raw_probs, self.current_labels[track_id]
        
        smoothed_probs = self._apply_ema(track_id, raw_probs)
        
        final_label = self._determine_label(track_id, smoothed_probs)
        
        return smoothed_probs, final_label
    
    def _apply_ema(self, 
                   track_id: int,
                   raw_probs: Dict[str, float]) -> Dict[str, float]:
        
        prev_probs = self.ema_states[track_id]
        smoothed = {}
        
        for label in raw_probs:
            if label in prev_probs:
                smoothed[label] = (self.ema_alpha * raw_probs[label] + 
                                  (1 - self.ema_alpha) * prev_probs[label])
            else:
                smoothed[label] = raw_probs[label]
        
        self.ema_states[track_id] = smoothed
        return smoothed
    
    def _determine_label(self,
                         track_id: int,
                         smoothed_probs: Dict[str, float]) -> str:
        
        current_label = self.current_labels.get(track_id, "neutral")
        new_label = max(smoothed_probs, key=smoothed_probs.get)
        
        if new_label == current_label:
            self.label_hold_counts[track_id] = 0
            self.pending_labels.pop(track_id, None)
            return current_label
        
        if track_id in self.pending_labels:
            if self.pending_labels[track_id] == new_label:
                self.label_hold_counts[track_id] += 1
                
                if self.label_hold_counts[track_id] >= self.min_hold_frames:
                    prob_diff = smoothed_probs[new_label] - smoothed_probs[current_label]
                    
                    if prob_diff >= self.switch_delta:
                        self.current_labels[track_id] = new_label
                        self.label_hold_counts[track_id] = 0
                        self.pending_labels.pop(track_id)
                        logger.debug(f"Track {track_id}: Label switched from {current_label} to {new_label}")
                        return new_label
            else:
                self.pending_labels[track_id] = new_label
                self.label_hold_counts[track_id] = 1
        else:
            self.pending_labels[track_id] = new_label
            self.label_hold_counts[track_id] = 1
        
        return current_label
    
    def reset_track(self, track_id: int):
        self.ema_states.pop(track_id, None)
        self.current_labels.pop(track_id, None)
        self.label_hold_counts.pop(track_id, None)
        self.pending_labels.pop(track_id, None)
    
    def get_smooth_probs(self, track_id: int) -> Optional[Dict[str, float]]:
        return self.ema_states.get(track_id)
    
    def get_current_label(self, track_id: int) -> str:
        return self.current_labels.get(track_id, "neutral")


class TemporalSmoother:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window_size))
    
    def add_sample(self, track_id: int, probs: Dict[str, float]):
        self.history[track_id].append(probs)
    
    def get_averaged(self, track_id: int) -> Optional[Dict[str, float]]:
        if track_id not in self.history or len(self.history[track_id]) == 0:
            return None
        
        samples = list(self.history[track_id])
        averaged = {}
        
        labels = samples[0].keys()
        for label in labels:
            values = [s[label] for s in samples if label in s]
            averaged[label] = np.mean(values) if values else 0.0
        
        total = sum(averaged.values())
        if total > 0:
            for label in averaged:
                averaged[label] /= total
        
        return averaged


class LabelStabilizer:
    def __init__(self,
                 confidence_threshold: float = 0.5,
                 stability_frames: int = 3):
        
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        self.label_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=stability_frames))
    
    def stabilize(self,
                  track_id: int,
                  label: str,
                  confidence: float) -> str:
        
        if confidence < self.confidence_threshold:
            label = "neutral"
        
        self.label_history[track_id].append(label)
        
        if len(self.label_history[track_id]) < self.stability_frames:
            return label
        
        labels = list(self.label_history[track_id])
        most_common = max(set(labels), key=labels.count)
        
        if labels.count(most_common) >= self.stability_frames - 1:
            return most_common
        
        return label