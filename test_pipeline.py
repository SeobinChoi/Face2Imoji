import sys
import numpy as np
import cv2
from pathlib import Path

# Add app to path for imports
sys.path.append(str(Path(__file__).parent))

from app.utils.data_types import Frame, PersonTrack, Landmark
from app.pose.detector import MultiPoseDetector
from app.face.detector import FaceDetector
from app.emotion.recognizer import EmotionRecognizer
from app.viz.renderer import Renderer, VisualizationConfig


def test_data_structures():
    print("Testing data structures...")
    
    # Test Landmark
    landmark = Landmark(x=100.0, y=200.0, conf=0.9)
    assert landmark.x == 100.0
    assert landmark.y == 200.0
    
    # Test Frame
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = Frame(frame_id=1, ts_ms=1000, image_bgr=test_image)
    assert frame.width == 640
    assert frame.height == 480
    
    print("✓ Data structures working correctly")


def test_mock_emotion_recognition():
    print("Testing emotion recognition...")
    
    recognizer = EmotionRecognizer(backend="MOCK")
    
    # Test with mock face crop
    face_crop = np.random.random((96, 96)) * 255
    face_crop = face_crop.astype(np.uint8)
    
    result = recognizer.recognize_single(face_crop)
    
    assert isinstance(result, dict)
    assert len(result) == 7  # Should have 7 emotion labels
    assert sum(result.values()) > 0.99  # Should sum to ~1.0 (normalized probabilities)
    
    print("✓ Emotion recognition working correctly")


def test_visualization():
    print("Testing visualization...")
    
    config = VisualizationConfig()
    renderer = Renderer(config=config)
    
    # Create test frame
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = Frame(frame_id=1, ts_ms=1000, image_bgr=test_image)
    
    # Create test track
    keypoints = [Landmark(x=320, y=240, conf=0.9) for _ in range(33)]
    track = PersonTrack(
        id=1,
        body_center=(320, 240),
        bbox_body=(280, 200, 80, 120),
        keypoints=keypoints,
        emotion_label="happy",
        emotion_prob={"happy": 0.8, "neutral": 0.2}
    )
    
    rendered = renderer.render(frame, [track])
    
    assert rendered.shape == test_image.shape
    assert not np.array_equal(rendered, test_image)  # Should be different from original
    
    print("✓ Visualization working correctly")


def test_components_import():
    print("Testing component imports...")
    
    try:
        from app.video_io.capture import VideoCapture, VideoDisplay
        from app.track.tracker import IoUTracker
        from app.smooth.smoother import EmotionSmoother
        from app.utils.performance import PerformanceMonitor
        print("✓ All components imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True


def test_config_loading():
    print("Testing configuration loading...")
    
    import yaml
    
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['video', 'pose', 'face', 'emotion', 'smooth', 'track', 'viz']
        for section in required_sections:
            assert section in config, f"Missing config section: {section}"
        
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Config loading error: {e}")
        return False


def main():
    print("Running pipeline tests...\n")
    
    tests = [
        test_data_structures,
        test_components_import,
        test_config_loading,
        test_mock_emotion_recognition,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = test()
            if result is not False:
                passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
        print()
    
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to run.")
        print("\nTo start the pipeline:")
        print("  python main.py")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())