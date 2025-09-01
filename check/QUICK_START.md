# Quick Start Guide

## 🚀 Immediate Usage

### 1. Test Setup (Recommended First Step)
```bash
python test_setup.py
```

### 2. Process All Videos (Default: 2 minutes each, nano model)
```bash
python yolo_player_detector.py
```

### 3. Process Single Video with Different Model
```bash
# Small model
python yolo_player_detector.py --single-video "../raw_videos/Aditi Narayan ｜ Matchplay.mp4" --model-size s

# Medium model  
python yolo_player_detector.py --single-video "../raw_videos/Aditi Narayan ｜ Matchplay.mp4" --model-size m

# Large model
python yolo_player_detector.py --single-video "../raw_videos/Aditi Narayan ｜ Matchplay.mp4" --model-size l
```

### 4. Compare All Model Sizes
```bash
python run_all_models.py
```

## 📁 What Gets Created

- **`annotated_videos/`** - Output directory with annotated videos
- **`yolov8n.pt`** - Nano model (fastest, lower accuracy)
- **`yolov8s.pt`** - Small model (balanced)
- **`yolov8m.pt`** - Medium model (better accuracy, slower)
- **`yolov8l.pt`** - Large model (best accuracy, slowest)

## ⚡ Model Performance Guide

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| **n** (nano) | ⚡⚡⚡ | ⭐⭐ | Quick testing, development |
| **s** (small) | ⚡⚡ | ⭐⭐⭐ | Good balance, production |
| **m** (medium) | ⚡ | ⭐⭐⭐⭐ | Better accuracy needed |
| **l** (large) | 🐌 | ⭐⭐⭐⭐⭐ | Best accuracy, research |

## 🔧 Key Features

- ✅ **CLI Model Selection**: `--model-size n/s/m/l`
- ✅ **Duration Control**: `--duration 2` (default: 2 minutes)
- ✅ **Single Video**: `--single-video path/to/video.mp4`
- ✅ **Batch Processing**: Processes all videos in `../raw_videos/`
- ✅ **Player Detection**: Specifically detects tennis players (person class)
- ✅ **Confidence Threshold**: 0.3 (adjustable in code)

## 📊 Output Format

Videos are saved as: `{original_name}_yolo_{model_size}.mp4`

Example: `Aditi Narayan ｜ Matchplay_yolo_n.mp4`

## 🎯 What It Detects

- **Primary Target**: Tennis players (person class)
- **Bounding Boxes**: Green rectangles around detected players
- **Confidence Scores**: Displayed above each detection
- **Filtering**: Only shows detections with >30% confidence

## 🚨 Troubleshooting

- **Model Download**: Models auto-download if not present
- **Memory Issues**: Use smaller models (n, s) for large videos
- **CUDA**: Automatically uses GPU if available, falls back to CPU
- **Video Access**: Ensure videos are in `../raw_videos/` directory

## 📚 Full Documentation

See `README.md` for complete usage details and examples.
