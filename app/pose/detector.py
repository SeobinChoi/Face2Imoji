import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple
import logging

from app.utils.data_types import Frame, PersonPose, Landmark

logger = logging.getLogger(__name__)


class PoseDetector:
    def __init__(self,
                 num_poses: int = 6,
                 min_detection_conf: float = 0.5,
                 min_tracking_conf: float = 0.5,
                 model_complexity: int = 1):
        
        self.num_poses = num_poses
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        
        logger.info(f"PoseDetector initialized: max {num_poses} poses, "
                   f"detection conf={min_detection_conf}, tracking conf={min_tracking_conf}")
    
    def detect(self, frame: Frame) -> List[PersonPose]:
        image_rgb = frame.get_rgb()
        
        results = self.pose.process(image_rgb)
        
        poses = []
        
        if results.pose_landmarks:
            keypoints = self._convert_landmarks(results.pose_landmarks, frame.width, frame.height)
            
            pose = PersonPose(
                keypoints=keypoints,
                score=self._calculate_pose_score(keypoints)
            )
            
            pose.bbox_body = pose.compute_body_bbox()
            
            poses.append(pose)
        
        return poses[:self.num_poses]
    
    def _convert_landmarks(self, 
                          landmarks,
                          width: int,
                          height: int) -> List[Landmark]:
        keypoints = []
        
        for lm in landmarks.landmark:
            keypoint = Landmark(
                x=lm.x * width,
                y=lm.y * height,
                z=lm.z if hasattr(lm, 'z') else None,
                conf=lm.visibility if hasattr(lm, 'visibility') else 1.0
            )
            keypoints.append(keypoint)
        
        return keypoints
    
    def _calculate_pose_score(self, keypoints: List[Landmark]) -> float:
        if not keypoints:
            return 0.0
        
        confidences = [kp.conf for kp in keypoints if kp.conf > 0]
        if not confidences:
            return 0.0
        
        return np.mean(confidences)
    
    def close(self):
        if self.pose:
            self.pose.close()


class MultiPoseDetector:
    def __init__(self,
                 num_poses: int = 6,
                 min_detection_conf: float = 0.5,
                 min_tracking_conf: float = 0.5):
        
        self.num_poses = num_poses
        self.min_detection_conf = min_detection_conf
        
        import mediapipe as mp
        self.mp_holistic = mp.solutions.holistic
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        
        self.previous_poses = []
        
        logger.info(f"MultiPoseDetector initialized for up to {num_poses} poses")
    
    def detect(self, frame: Frame) -> List[PersonPose]:
        image_rgb = frame.get_rgb()
        
        results = self.holistic.process(image_rgb)
        
        poses = []
        
        if results.pose_landmarks:
            keypoints = self._convert_landmarks(results.pose_landmarks, frame.width, frame.height)
            
            pose = PersonPose(
                keypoints=keypoints,
                score=self._calculate_pose_score(keypoints)
            )
            
            pose.bbox_body = self._compute_body_bbox_pixels(keypoints)
            
            poses.append(pose)
        
        self.previous_poses = poses
        return poses[:self.num_poses]
    
    def _convert_landmarks(self,
                          landmarks,
                          width: int,
                          height: int) -> List[Landmark]:
        keypoints = []
        
        for lm in landmarks.landmark:
            keypoint = Landmark(
                x=lm.x * width,
                y=lm.y * height,
                z=lm.z if hasattr(lm, 'z') else None,
                conf=lm.visibility if hasattr(lm, 'visibility') else 1.0
            )
            keypoints.append(keypoint)
        
        return keypoints
    
    def _compute_body_bbox_pixels(self, keypoints: List[Landmark]) -> Tuple[float, float, float, float]:
        if not keypoints:
            return (0, 0, 0, 0)
        
        valid_points = [(kp.x, kp.y) for kp in keypoints if kp.conf > 0.3]
        
        if not valid_points:
            return (0, 0, 0, 0)
        
        xs = [p[0] for p in valid_points]
        ys = [p[1] for p in valid_points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        padding = 20
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        width = max_x - min_x + 2 * padding
        height = max_y - min_y + 2 * padding
        
        return (min_x, min_y, width, height)
    
    def _calculate_pose_score(self, keypoints: List[Landmark]) -> float:
        if not keypoints:
            return 0.0
        
        key_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
        
        scores = []
        for idx in key_indices:
            if idx < len(keypoints):
                scores.append(keypoints[idx].conf)
        
        if not scores:
            return 0.0
        
        return np.mean(scores)
    
    def close(self):
        if self.holistic:
            self.holistic.close()