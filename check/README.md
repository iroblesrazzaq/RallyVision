# YOLO Player Detector for Tennis Videos

This tool uses YOLO object detection to recognize and annotate players in tennis videos.

## Features

- **Multiple Model Sizes**: Choose from nano (n), small (s), medium (m), or large (l) YOLO models
- **Batch Processing**: Process all videos in a directory automatically
- **Duration Control**: Process only the first 2 minutes (or custom duration) of each video
- **Player Detection**: Specifically detects and annotates person objects (tennis players)
- **Video Annotation**: Draws bounding boxes and confidence scores on detected players

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the YOLO model files in the parent directory:
   - `yolov8n.pt` (nano) - included
   - `yolov8s.pt` (small) - will download if not present
   - `yolov8m.pt` (medium) - will download if not present
   - `yolov8l.pt` (large) - will download if not present

## Usage

### Process All Videos (Default)

Process all videos in the raw_videos directory with nano model:

```bash
python yolo_player_detector.py
```

### Specify Model Size

Use a different YOLO model size:

```bash
# Small model
python yolo_player_detector.py --model-size s

# Medium model  
python yolo_player_detector.py --model-size m

# Large model
python yolo_player_detector.py --model-size l
```

### Custom Input/Output Directories

```bash
python yolo_player_detector.py \
    --input-dir /path/to/videos \
    --output-dir /path/to/output \
    --model-size s
```

### Process Single Video

```bash
python yolo_player_detector.py \
    --single-video /path/to/video.mp4 \
    --model-size m
```

### Custom Duration

Process more than 2 minutes (default):

```bash
python yolo_player_detector.py --duration 5
```

## Command Line Arguments

- `--model-size, -m`: YOLO model size (n, s, m, l) - default: n
- `--input-dir, -i`: Input directory containing videos - default: ../raw_videos
- `--output-dir, -o`: Output directory for annotated videos - default: ./annotated_videos
- `--duration, -d`: Duration to process in minutes - default: 2
- `--single-video, -v`: Process single video file instead of directory

## Output

- Annotated videos are saved to the output directory
- Filename format: `{original_name}_yolo_{model_size}.mp4`
- Green bounding boxes around detected players
- Confidence scores displayed above each detection

## Performance Notes

- **Nano (n)**: Fastest, good for testing, lower accuracy
- **Small (s)**: Good balance of speed and accuracy
- **Medium (m)**: Better accuracy, slower processing
- **Large (l)**: Best accuracy, slowest processing

## Example Output

```
Using device: cpu
Loading existing model: yolov8n.pt
Model loaded successfully: n
Found 20 video files
Processing video: ../raw_videos/Aditi Narayan ｜ Matchplay.mp4
Video: 30 FPS, 1920x1080, 3690 frames
Processing 3600 frames (2 minutes)
Processed 300/3600 frames
Processed 600/3600 frames
...
Annotated video saved to: ./annotated_videos/Aditi Narayan ｜ Matchplay_yolo_n.mp4
Successfully processed: Aditi Narayan ｜ Matchplay.mp4
```

## Notes

- The tool automatically downloads YOLO models if they don't exist locally
- Processing time depends on video length, resolution, and model size
- Person detection uses a confidence threshold of 0.3
- Only the first 2 minutes (or specified duration) of each video is processed
