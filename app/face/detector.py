import mediapipe as mp
import numpy as np
import cv2
from typing import List, Optional, Tuple
import logging

from app.utils.data_types import Frame, Detection

logger = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self,
                 min_conf: float = 0.6,
                 nms_iou: float = 0.4,
                 max_faces: int = 10,
                 crop_size: int = 96):
        
        self.min_conf = min_conf
        self.nms_iou = nms_iou
        self.max_faces = max_faces
        self.crop_size = crop_size
        
        self.mp_face_detection = mp.solutions.face_detection
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range (within 2 meters)
            min_detection_confidence=min_conf
        )
        
        logger.info(f"FaceDetector initialized: min_conf={min_conf}, "
                   f"max_faces={max_faces}, crop_size={crop_size}")
    
    def detect(self, frame: Frame) -> List[Detection]:
        image_rgb = frame.get_rgb()
        
        results = self.face_detection.process(image_rgb)
        
        detections = []
        
        if results.detections:
            for detection in results.detections[:self.max_faces]:
                bbox = self._get_bbox_from_detection(detection, frame.width, frame.height)
                score = detection.score[0] if detection.score else self.min_conf
                
                det = Detection(bbox=bbox, score=score)
                detections.append(det)
        
        detections = self._apply_nms(detections)
        
        return detections[:self.max_faces]
    
    def _get_bbox_from_detection(self, 
                                 detection,
                                 width: int,
                                 height: int) -> Tuple[float, float, float, float]:
        bbox = detection.location_data.relative_bounding_box
        
        x = bbox.xmin * width
        y = bbox.ymin * height
        w = bbox.width * width
        h = bbox.height * height
        
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)
        
        return (x, y, w, h)
    
    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        if len(detections) <= 1:
            return detections
        
        boxes = np.array([d.to_xyxy() for d in detections])
        scores = np.array([d.score for d in detections])
        
        indices = self._nms(boxes, scores, self.nms_iou)
        
        return [detections[i] for i in indices]
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def get_face_crops(self, 
                      frame: Frame,
                      detections: List[Detection],
                      grayscale: bool = True,
                      normalize: str = "zero_to_one") -> List[np.ndarray]:
        
        crops = []
        image = frame.image_bgr
        
        for det in detections:
            x, y, w, h = [int(v) for v in det.bbox]
            
            cx, cy = x + w // 2, y + h // 2
            size = max(w, h)
            half_size = size // 2
            
            x1 = max(0, cx - half_size)
            y1 = max(0, cy - half_size)
            x2 = min(frame.width, cx + half_size)
            y2 = min(frame.height, cy + half_size)
            
            crop = image[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            crop = cv2.resize(crop, (self.crop_size, self.crop_size))
            
            if grayscale:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            if normalize == "zero_to_one":
                crop = crop.astype(np.float32) / 255.0
            elif normalize == "minus_one_to_one":
                crop = (crop.astype(np.float32) / 127.5) - 1.0
            
            crops.append(crop)
        
        return crops
    
    def close(self):
        if self.face_detection:
            self.face_detection.close()


class BatchFaceProcessor:
    def __init__(self, 
                 detector: FaceDetector,
                 batch_size: int = 8):
        
        self.detector = detector
        self.batch_size = batch_size
        self.pending_faces = []
        
    def add_faces(self, 
                 frame: Frame,
                 detections: List[Detection],
                 track_ids: List[int]):
        
        crops = self.detector.get_face_crops(frame, detections)
        
        for crop, det, track_id in zip(crops, detections, track_ids):
            self.pending_faces.append({
                'crop': crop,
                'bbox': det.bbox,
                'track_id': track_id,
                'frame_id': frame.frame_id
            })
    
    def get_batch(self) -> Optional[Tuple[np.ndarray, List[int]]]:
        if len(self.pending_faces) < 1:
            return None
        
        batch_faces = self.pending_faces[:self.batch_size]
        self.pending_faces = self.pending_faces[self.batch_size:]
        
        crops = np.array([f['crop'] for f in batch_faces])
        track_ids = [f['track_id'] for f in batch_faces]
        
        if len(crops.shape) == 3:  # Grayscale
            crops = np.expand_dims(crops, axis=-1)
        
        return crops, track_ids
    
    def clear(self):
        self.pending_faces = []