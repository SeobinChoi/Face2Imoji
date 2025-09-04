import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

from app.utils.data_types import PersonTrack, PersonPose, Detection, TrackState, Landmark

logger = logging.getLogger(__name__)


class IoUTracker:
    def __init__(self,
                 iou_thresh: float = 0.3,
                 max_age: int = 10,
                 min_hits: int = 3):
        
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.tracks: Dict[int, PersonTrack] = {}
        self.next_id = 1
        self.frame_count = 0
        
        logger.info(f"IoUTracker initialized: iou_thresh={iou_thresh}, "
                   f"max_age={max_age}, min_hits={min_hits}")
    
    def update(self,
               poses: List[PersonPose],
               face_detections: List[Detection],
               ts_ms: int) -> List[PersonTrack]:
        
        self.frame_count += 1
        
        body_bboxes = [pose.bbox_body for pose in poses if pose.bbox_body]
        
        matched_track_ids, unmatched_det_ids, unmatched_track_ids = self._match_detections(
            body_bboxes,
            list(self.tracks.keys())
        )
        
        updated_tracks = []
        
        for det_idx, track_id in matched_track_ids:
            pose = poses[det_idx]
            track = self.tracks[track_id]
            
            track.keypoints = pose.keypoints
            track.bbox_body = pose.bbox_body
            track.body_center = self._compute_body_center(pose.keypoints)
            track.last_update_ms = ts_ms
            track.hits += 1
            track.age = 0
            
            if track.hits >= self.min_hits:
                track.state = TrackState.CONFIRMED
            
            updated_tracks.append(track)
        
        for det_idx in unmatched_det_ids:
            pose = poses[det_idx]
            new_track = self._create_track(pose, ts_ms)
            self.tracks[new_track.id] = new_track
            updated_tracks.append(new_track)
        
        for track_id in unmatched_track_ids:
            track = self.tracks[track_id]
            track.age += 1
            
            if track.age > self.max_age:
                track.state = TrackState.DELETED
            elif track.state == TrackState.CONFIRMED:
                updated_tracks.append(track)
        
        face_matches = self._match_faces_to_tracks(updated_tracks, face_detections)
        for track, face_bbox in face_matches:
            track.face_bbox = face_bbox
        
        self.tracks = {t.id: t for t in updated_tracks if t.state != TrackState.DELETED}
        
        return [t for t in updated_tracks if t.state == TrackState.CONFIRMED]
    
    def _match_detections(self,
                         detections: List[Tuple[float, float, float, float]],
                         track_ids: List[int]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        
        if not detections or not track_ids:
            return [], list(range(len(detections))), track_ids
        
        iou_matrix = np.zeros((len(detections), len(track_ids)))
        
        for d_idx, det_bbox in enumerate(detections):
            for t_idx, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                iou_matrix[d_idx, t_idx] = self._compute_iou(det_bbox, track.bbox_body)
        
        matched_indices = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(track_ids)
        
        if iou_matrix.size > 0:
            for _ in range(min(len(detections), len(track_ids))):
                if iou_matrix.size == 0:
                    break
                
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                
                if iou_matrix[max_idx] < self.iou_thresh:
                    break
                
                det_idx, track_idx = max_idx
                track_id = track_ids[track_idx]
                
                matched_indices.append((det_idx, track_id))
                unmatched_dets.discard(det_idx)
                unmatched_tracks.discard(track_id)
                
                iou_matrix[det_idx, :] = -1
                iou_matrix[:, track_idx] = -1
        
        return matched_indices, list(unmatched_dets), list(unmatched_tracks)
    
    def _compute_iou(self, 
                    bbox1: Tuple[float, float, float, float],
                    bbox2: Tuple[float, float, float, float]) -> float:
        
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    def _create_track(self, pose: PersonPose, ts_ms: int) -> PersonTrack:
        track = PersonTrack(
            id=self.next_id,
            body_center=self._compute_body_center(pose.keypoints),
            bbox_body=pose.bbox_body,
            keypoints=pose.keypoints,
            last_update_ms=ts_ms,
            state=TrackState.TENTATIVE,
            hits=1
        )
        
        self.next_id += 1
        return track
    
    def _compute_body_center(self, keypoints: List[Landmark]) -> Tuple[float, float]:
        shoulder_indices = [11, 12]
        hip_indices = [23, 24]
        
        points = []
        for idx in shoulder_indices + hip_indices:
            if idx < len(keypoints) and keypoints[idx].conf > 0.3:
                points.append((keypoints[idx].x, keypoints[idx].y))
        
        if not points:
            valid_points = [(kp.x, kp.y) for kp in keypoints if kp.conf > 0.3]
            if valid_points:
                center_x = sum(p[0] for p in valid_points) / len(valid_points)
                center_y = sum(p[1] for p in valid_points) / len(valid_points)
                return (center_x, center_y)
            return (0, 0)
        
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        return (center_x, center_y)
    
    def _match_faces_to_tracks(self,
                               tracks: List[PersonTrack],
                               face_detections: List[Detection]) -> List[Tuple[PersonTrack, Tuple]]:
        
        if not tracks or not face_detections:
            return []
        
        matches = []
        used_faces = set()
        
        for track in tracks:
            if track.body_center == (0, 0):
                continue
            
            best_face_idx = -1
            best_distance = float('inf')
            
            for face_idx, face_det in enumerate(face_detections):
                if face_idx in used_faces:
                    continue
                
                face_center = face_det.center
                
                distance = np.sqrt(
                    (track.body_center[0] - face_center[0]) ** 2 +
                    (track.body_center[1] - face_center[1]) ** 2
                )
                
                body_diagonal = np.sqrt(track.bbox_body[2] ** 2 + track.bbox_body[3] ** 2)
                max_distance = body_diagonal * 0.7
                
                if distance < best_distance and distance < max_distance:
                    best_distance = distance
                    best_face_idx = face_idx
            
            if best_face_idx >= 0:
                matches.append((track, face_detections[best_face_idx].bbox))
                used_faces.add(best_face_idx)
        
        return matches
    
    def get_track_by_id(self, track_id: int) -> Optional[PersonTrack]:
        return self.tracks.get(track_id)
    
    def get_all_tracks(self) -> List[PersonTrack]:
        return [t for t in self.tracks.values() if t.state == TrackState.CONFIRMED]
    
    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
        logger.info("Tracker reset")