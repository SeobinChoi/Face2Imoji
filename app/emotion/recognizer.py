import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from collections import deque
import cv2

logger = logging.getLogger(__name__)


class EmotionRecognizer:
    def __init__(self,
                 backend: str = "FER",
                 labels: List[str] = None,
                 every_n_frames: int = 3,
                 batch_size: int = 8):
        
        self.backend = backend
        self.labels = labels or ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        self.every_n_frames = every_n_frames
        self.batch_size = batch_size
        
        self.frame_counter = 0
        self.model = None
        
        self._init_model()
        
        logger.info(f"EmotionRecognizer initialized: backend={backend}, "
                   f"labels={self.labels}, every_n_frames={every_n_frames}")
    
    def _init_model(self):
        if self.backend == "FER":
            try:
                from fer import FER
                self.model = FER(mtcnn=False)
                logger.info("FER model loaded successfully")
            except ImportError:
                logger.warning("FER not available, using mock emotion recognizer")
                self.model = MockEmotionModel(self.labels)
        else:
            self.model = MockEmotionModel(self.labels)
    
    def should_process(self) -> bool:
        self.frame_counter += 1
        return self.frame_counter % self.every_n_frames == 0
    
    def recognize_batch(self, face_crops: np.ndarray) -> List[Dict[str, float]]:
        if self.model is None:
            return [self._get_default_emotions() for _ in face_crops]
        
        if isinstance(self.model, MockEmotionModel):
            return [self.model.predict(crop) for crop in face_crops]
        
        results = []
        for crop in face_crops:
            if len(crop.shape) == 2:  # Grayscale
                crop = cv2.cvtColor((crop * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            elif len(crop.shape) == 3 and crop.shape[-1] == 1:
                crop = cv2.cvtColor((crop * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
            try:
                emotions = self.model.detect_emotions(crop)
                if emotions and len(emotions) > 0:
                    emotion_probs = emotions[0]['emotions']
                    results.append(self._normalize_emotions(emotion_probs))
                else:
                    results.append(self._get_default_emotions())
            except Exception as e:
                logger.error(f"Error in emotion recognition: {e}")
                results.append(self._get_default_emotions())
        
        return results
    
    def recognize_single(self, face_crop: np.ndarray) -> Dict[str, float]:
        results = self.recognize_batch(np.array([face_crop]))
        return results[0] if results else self._get_default_emotions()
    
    def _normalize_emotions(self, emotions: Dict[str, float]) -> Dict[str, float]:
        normalized = {}
        for label in self.labels:
            if label in emotions:
                normalized[label] = emotions[label]
            else:
                normalized[label] = 0.0
        
        total = sum(normalized.values())
        if total > 0:
            for label in normalized:
                normalized[label] /= total
        
        return normalized
    
    def _get_default_emotions(self) -> Dict[str, float]:
        emotions = {label: 0.0 for label in self.labels}
        emotions["neutral"] = 1.0
        return emotions
    
    def adjust_processing_rate(self, queue_length: int):
        if queue_length > self.batch_size * 2:
            self.every_n_frames = min(self.every_n_frames + 2, 10)
            logger.warning(f"Emotion processing overloaded, reducing rate to every {self.every_n_frames} frames")
        elif queue_length < self.batch_size and self.every_n_frames > 3:
            self.every_n_frames = max(self.every_n_frames - 1, 3)


class MockEmotionModel:
    def __init__(self, labels: List[str]):
        self.labels = labels
        self.state_counter = 0
    
    def predict(self, face_crop: np.ndarray) -> Dict[str, float]:
        self.state_counter += 1
        
        mean_brightness = np.mean(face_crop)
        
        emotions = {label: 0.1 for label in self.labels}
        
        if mean_brightness > 0.6:
            emotions["happy"] = 0.4 + np.random.random() * 0.3
        elif mean_brightness < 0.3:
            emotions["sad"] = 0.4 + np.random.random() * 0.3
        else:
            emotions["neutral"] = 0.5 + np.random.random() * 0.2
        
        if self.state_counter % 30 < 10:
            emotions["surprise"] = 0.3 + np.random.random() * 0.2
        
        total = sum(emotions.values())
        for label in emotions:
            emotions[label] /= total
        
        return emotions


class EmotionQueue:
    def __init__(self, max_size: int = 100):
        self.queue = deque(maxlen=max_size)
        self.results: Dict[int, Dict[str, float]] = {}
    
    def add(self, track_id: int, face_crop: np.ndarray, frame_id: int):
        self.queue.append({
            'track_id': track_id,
            'face_crop': face_crop,
            'frame_id': frame_id
        })
    
    def process_batch(self, recognizer: EmotionRecognizer) -> Dict[int, Dict[str, float]]:
        if not self.queue:
            return {}
        
        batch_size = min(recognizer.batch_size, len(self.queue))
        batch_items = [self.queue.popleft() for _ in range(batch_size)]
        
        face_crops = np.array([item['face_crop'] for item in batch_items])
        emotions = recognizer.recognize_batch(face_crops)
        
        results = {}
        for item, emotion in zip(batch_items, emotions):
            results[item['track_id']] = emotion
            self.results[item['track_id']] = emotion
        
        return results
    
    def get_result(self, track_id: int) -> Optional[Dict[str, float]]:
        return self.results.get(track_id)
    
    def clear(self):
        self.queue.clear()
        self.results.clear()
    
    @property
    def size(self) -> int:
        return len(self.queue)