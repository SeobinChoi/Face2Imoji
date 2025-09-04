# Real-time Multi-Pose Emotion Detection Pipeline

A high-performance real-time computer vision pipeline that combines multi-person pose detection, face recognition, and emotion analysis with smooth emoji overlays.

## Features

- **Multi-person pose detection** using MediaPipe Pose
- **Real-time face detection and tracking** with persistent IDs
- **Emotion recognition** with 7 emotion classes (happy, sad, angry, surprise, fear, disgust, neutral)
- **Smooth emotion transitions** using EMA filtering and label stability
- **Interactive visualization** with skeleton overlay, emoji balloons, and HUD
- **Adaptive performance** with automatic downscaling for consistent 15+ FPS
- **720p real-time processing** on consumer hardware

## Architecture

```
Frame → Pose Detection → Face Detection → Tracking → Emotion Recognition
                    ↓
              Smoothing → Visualization → Display
```

### Pipeline Components

1. **Video I/O**: Camera capture with adaptive resolution scaling
2. **Pose Detection**: MediaPipe-based multi-person pose estimation
3. **Face Detection**: Short-range face detection with NMS
4. **Tracking**: IoU-based person tracking with ID persistence
5. **Emotion Recognition**: Batch-processed emotion classification
6. **Smoothing**: EMA-based probability smoothing and label stabilization
7. **Visualization**: Skeleton rendering with emoji balloons and HUD

## Installation

```bash
pip install -r requirements.txt
python create_logo.py  # Create placeholder logo
```

## Usage

```bash
python main.py [config.yaml]
```

### Controls
- `Q`: Quit application
- `R`: Reset tracking (clear all person IDs)
- `E`: Toggle emotion processing

## Configuration

Edit `config.yaml` to customize:

- **Video settings**: Resolution, FPS, camera device
- **Detection thresholds**: Pose and face confidence levels
- **Tracking parameters**: IoU thresholds, track persistence
- **Emotion settings**: Processing frequency, smoothing parameters
- **Visualization**: Colors, sizes, HUD layout

## Performance

Target performance on typical hardware:
- **720p @ 20-25 FPS** (3-5 people)
- **540p @ 15+ FPS** (6+ people, auto-downscaling)
- **Memory usage**: ~200-400MB
- **Latency**: <120ms end-to-end

### Optimization Features
- Adaptive processing rates based on load
- Batch emotion recognition every N frames
- Automatic resolution downscaling
- Memory usage monitoring

## File Structure

```
app/
├── video_io/     # Camera capture and display
├── pose/         # MediaPipe pose detection
├── face/         # Face detection and cropping
├── track/        # Person tracking with IoU matching
├── emotion/      # Emotion recognition (FER/ONNX)
├── smooth/       # EMA smoothing and label stability
├── viz/          # Rendering and visualization
└── utils/        # Performance monitoring and data types

config.yaml       # Configuration parameters
main.py          # Main pipeline orchestration
```

## Emotion Classes

- 😄 **happy**: Smiles and positive expressions
- 😐 **neutral**: Calm, default expression
- 😲 **surprise**: Wide eyes, open mouth
- 😢 **sad**: Frowning, downturned mouth
- 😡 **angry**: Furrowed brow, tense expression
- 😨 **fear**: Wide eyes, tense features
- 🤢 **disgust**: Nose wrinkled, mouth corners down

## Technical Details

### Smoothing Algorithm
- **EMA filtering**: `p_t = α·p_raw + (1-α)·p_{t-1}` with α=0.4
- **Label switching**: Requires confidence delta ≥0.15 + minimum 5 frame hold
- **Prevents flickering** while maintaining responsiveness

### Tracking Algorithm
- **IoU-based matching** between frame-to-frame detections
- **Minimum 3 hits** before confirming track
- **10 frame persistence** after detection loss
- **Face-body association** using spatial proximity

### Adaptive Performance
- **Processing frequency**: Emotion every 3 frames, face every 1-2 frames
- **Queue management**: Automatic rate adjustment under load
- **Resolution scaling**: 1280×720 → 960×540 → 720×405 as needed

## Requirements

- Python 3.8+
- OpenCV 4.8+
- MediaPipe 0.10+
- NumPy 1.21+
- FER 22.5+ (or ONNXRuntime for ONNX backend)

## Demo Setup

1. **Lighting**: Consistent, front-facing light
2. **Camera angle**: Chest-level, capture full upper bodies
3. **Background**: Simple, non-cluttered
4. **Distance**: 1-3 meters from camera for optimal face detection

## Troubleshooting

- **Low FPS**: Check `min_fps` in config, enable auto-downscaling
- **Poor emotion accuracy**: Increase `min_conf` for face detection
- **Tracking issues**: Adjust `iou_thresh` and `max_age` parameters
- **Memory usage**: Reduce `batch_size` or enable more aggressive processing intervals