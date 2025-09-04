import yaml
import logging
import cv2
import sys
from pathlib import Path
from typing import Dict, Any

from app.video_io.capture import VideoCapture, VideoDisplay
from app.pose.detector import MultiPoseDetector
from app.face.detector import FaceDetector, BatchFaceProcessor
from app.track.tracker import IoUTracker
from app.emotion.recognizer import EmotionRecognizer, EmotionQueue
from app.smooth.smoother import EmotionSmoother
from app.viz.renderer import Renderer, VisualizationConfig
from app.utils.performance import PerformanceMonitor, AdaptiveProcessor, MemoryMonitor


class EmotionDetectionPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self._setup_logging()
        
        self.performance_monitor = PerformanceMonitor()
        self.adaptive_processor = AdaptiveProcessor(
            target_fps=self.config['video']['target_fps'] * 0.8,
            min_fps=self.config['video']['min_fps']
        )
        self.memory_monitor = MemoryMonitor()
        
        self._init_components()
        
        self.running = False
        
        logger = logging.getLogger(__name__)
        logger.info("EmotionDetectionPipeline initialized")
    
    def _setup_logging(self):
        level = getattr(logging, self.config['runtime']['log_level'])
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _init_components(self):
        
        self.video_capture = VideoCapture(
            device_index=self.config['video']['device_index'],
            target_size=(self.config['video']['width'], self.config['video']['height']),
            target_fps=self.config['video']['target_fps'],
            mirror=self.config['video']['mirror'],
            min_fps=self.config['video']['min_fps'],
            downscale_steps=self.config['video']['downscale_steps']
        )
        
        self.video_display = VideoDisplay()
        
        self.pose_detector = MultiPoseDetector(
            num_poses=self.config['pose']['num_poses'],
            min_detection_conf=self.config['pose']['min_detection_conf'],
            min_tracking_conf=self.config['pose']['min_tracking_conf']
        )
        
        self.face_detector = FaceDetector(
            min_conf=self.config['face']['min_conf'],
            nms_iou=self.config['face']['nms_iou'],
            max_faces=self.config['face']['max_faces'],
            crop_size=self.config['face']['crop_size']
        )
        
        self.batch_face_processor = BatchFaceProcessor(
            detector=self.face_detector,
            batch_size=self.config['emotion']['batch_size']
        )
        
        self.tracker = IoUTracker(
            iou_thresh=self.config['track']['iou_thresh'],
            max_age=self.config['track']['max_age'],
            min_hits=self.config['track']['min_hits']
        )
        
        self.emotion_recognizer = EmotionRecognizer(
            backend=self.config['emotion']['backend'],
            labels=self.config['emotion']['labels'],
            every_n_frames=self.config['emotion']['every_n_frames'],
            batch_size=self.config['emotion']['batch_size']
        )
        
        self.emotion_queue = EmotionQueue()
        
        self.emotion_smoother = EmotionSmoother(
            ema_alpha=self.config['smooth']['ema_alpha'],
            switch_delta=self.config['smooth']['switch_delta'],
            min_hold_frames=self.config['smooth']['min_hold_frames']
        )
        
        viz_config = VisualizationConfig(
            balloon_diameter_px=self.config['viz']['balloon']['diameter_px'],
            balloon_alpha=self.config['viz']['balloon']['alpha'],
            balloon_offset_y_px=self.config['viz']['balloon']['offset_y_px'],
            balloon_font_scale=self.config['viz']['balloon']['font_scale'],
            balloon_prob_digits=self.config['viz']['balloon']['prob_digits'],
            skeleton_line_thickness=self.config['viz']['skeleton']['line_thickness'],
            skeleton_point_radius=self.config['viz']['skeleton']['point_radius'],
            hud_pos=self.config['viz']['hud']['pos'],
            hud_alpha=self.config['viz']['hud']['alpha'],
            hud_font_scale=self.config['viz']['hud']['font_scale'],
            logo_path=self.config['viz']['logo']['path'],
            logo_alpha=self.config['viz']['logo']['alpha'],
            logo_pos=self.config['viz']['logo']['pos']
        )
        
        self.renderer = Renderer(
            config=viz_config,
            emoji_mapping=self.config['emoji_mapping']
        )
    
    def process_frame(self, frame):
        
        self.performance_monitor.start_stage("pose")
        poses = self.pose_detector.detect(frame)
        self.performance_monitor.end_stage("pose")
        
        face_detections = []
        if self.adaptive_processor.should_process("face"):
            self.performance_monitor.start_stage("face")
            face_detections = self.face_detector.detect(frame)
            self.performance_monitor.end_stage("face")
        
        self.performance_monitor.start_stage("tracking")
        tracks = self.tracker.update(poses, face_detections, frame.ts_ms)
        self.performance_monitor.end_stage("tracking")
        
        if self.emotion_recognizer.should_process():
            self.performance_monitor.start_stage("emotion")
            self._process_emotions(frame, tracks)
            self.performance_monitor.end_stage("emotion")
        
        batch_results = self.emotion_queue.process_batch(self.emotion_recognizer)
        self._update_track_emotions(tracks, batch_results)
        
        self.performance_monitor.start_stage("viz")
        rendered_frame = self.renderer.render(
            frame, tracks, max_render=self.config['runtime']['max_people_render']
        )
        self.performance_monitor.end_stage("viz")
        
        return rendered_frame
    
    def _process_emotions(self, frame, tracks):
        for track in tracks:
            if track.face_bbox is not None:
                crops = self.face_detector.get_face_crops(
                    frame,
                    [type('obj', (object,), {'bbox': track.face_bbox, 'score': 1.0})()],
                    grayscale=self.config['face']['grayscale'],
                    normalize=self.config['face']['normalize']
                )
                
                if crops:
                    self.emotion_queue.add(track.id, crops[0], frame.frame_id)
    
    def _update_track_emotions(self, tracks, batch_results):
        for track in tracks:
            if track.id in batch_results:
                raw_probs = batch_results[track.id]
                
                smoothed_probs, final_label = self.emotion_smoother.smooth(
                    track.id, raw_probs
                )
                
                track.emotion_prob = smoothed_probs
                track.emotion_label = final_label
            else:
                existing_result = self.emotion_queue.get_result(track.id)
                if existing_result:
                    smoothed_probs, final_label = self.emotion_smoother.smooth(
                        track.id, existing_result
                    )
                    track.emotion_prob = smoothed_probs
                    track.emotion_label = final_label
    
    def run(self):
        self.running = True
        logger = logging.getLogger(__name__)
        logger.info("Starting emotion detection pipeline")
        
        frame_count = 0
        
        try:
            for frame in self.video_capture.stream_frames():
                if not self.running:
                    break
                
                self.performance_monitor.start_stage("capture")
                self.performance_monitor.end_stage("capture")
                
                rendered_frame = self.process_frame(frame)
                
                result = self.video_display.show_frame(rendered_frame)
                if result is False:
                    break
                elif result == 'reset':
                    self.tracker.reset()
                    self.emotion_queue.clear()
                    logger.info("Pipeline reset")
                elif result == 'toggle_emotion':
                    pass
                
                self.performance_monitor.frame_complete()
                
                current_fps = self.performance_monitor.get_current_fps()
                self.adaptive_processor.adapt(current_fps, self.emotion_queue.size)
                
                frame_count += 1
                if frame_count % 300 == 0:  # Every 10 seconds at 30fps
                    self.performance_monitor.log_stats()
                
                if self.memory_monitor.should_check():
                    self.memory_monitor.log_memory()
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        finally:
            self._cleanup()
    
    def stop(self):
        self.running = False
    
    def _cleanup(self):
        logger = logging.getLogger(__name__)
        logger.info("Cleaning up pipeline")
        
        self.video_capture.release()
        self.video_display.destroy()
        
        if hasattr(self.pose_detector, 'close'):
            self.pose_detector.close()
        if hasattr(self.face_detector, 'close'):
            self.face_detector.close()


def main():
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config.yaml"
    
    if not Path(config_path).exists():
        print(f"Config file {config_path} not found")
        return 1
    
    try:
        pipeline = EmotionDetectionPipeline(config_path)
        pipeline.run()
    except Exception as e:
        print(f"Failed to start pipeline: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())