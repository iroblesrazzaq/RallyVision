# Enhanced Feature Visualizer

This tool validates engineered features by mapping feature vectors to original video frames, combining preprocessed data with feature vectors for comprehensive validation.

## 🎯 Purpose

The enhanced feature visualizer solves the critical issue of validating that feature engineering is working correctly by:

1. **Mapping feature vectors to video frames** despite different frame rates
2. **Displaying engineered features** (velocity, acceleration, etc.) directly from feature vectors
3. **Validating data integrity** by cross-referencing preprocessed and feature data
4. **Providing visual confirmation** that feature engineering produces correct results

## 📁 Data Flow

```
Original Video (25-30 FPS)
        ↓
Preprocessed Data (15 FPS with -100 skipped frames)
        ↓
Feature Engineering (Creates 360-element vectors from annotated frames only)
        ↓
Enhanced Feature Visualizer (Maps feature vectors back to original video frames)
```

## 🚀 Usage

### List Available Videos
```bash
python3 run_enhanced_visualization.py --list
```

### Run Visualization for Specific Video
```bash
python3 run_enhanced_visualization.py --video-pattern "Satoru Nakajima" --start-time 0 --duration 30
```

### Run Visualization for All Videos
```bash
python3 run_enhanced_visualization.py --all --start-time 0 --duration 15
```

### Command Line Options
- `--video-pattern` or `-p`: Pattern to match video names
- `--start-time` or `-s`: Start time in seconds (default: 0)
- `--duration` or `-d`: Duration in seconds (default: 30)
- `--output-dir` or `-o`: Output directory (default: sanity_check_clips)
- `--list` or `-l`: List available videos
- `--all` or `-a`: Process all available videos

## ✅ Features Displayed

### Player Features (Per Player)
- **Bounding Box**: Green rectangle around player
- **Centroid**: Red dot at player center
- **Velocity Vector**: Cyan arrow showing movement direction
- **Acceleration Vector**: Magenta arrow showing acceleration direction
- **Speed Magnitude**: Numeric value of speed
- **Acceleration Magnitude**: Numeric value of acceleration magnitude

### Frame Information
- **Point Status**: "IN POINT" (cyan) or "NOT IN POINT" (red)
- **Skipped Frames**: "SKIPPED FRAME" (gray) for temporally downsampled frames
- **Court Mask**: Green overlay showing detected court area

## 🔧 Technical Implementation

### Data Mapping
The visualizer correctly maps between different data sources:

1. **Preprocessed Data**: Provides frame timing and court mask
2. **Feature Vectors**: Contains engineered features (360 elements per frame)
3. **Original Video**: Provides visual context

### Feature Vector Structure (360 elements)
- **Player 1 (Elements 0-179)**: Near player features
- **Player 2 (Elements 180-359)**: Far player features

Each player has:
- Presence flag (1.0 = present, -1.0 = absent)
- Bounding box coordinates [x1, y1, x2, y2]
- Centroid coordinates [cx, cy]
- Velocity vector [vx, vy]
- Acceleration vector [ax, ay]
- Speed magnitude
- Acceleration magnitude
- Keypoint data (positions, confidences, velocities, accelerations)
- Limb lengths

### Validation Process
1. **Data Integrity Check**: Verifies that annotated frame count matches feature vector count
2. **Target Alignment**: Ensures target values (0/1) align between preprocessed and feature data
3. **Feature Extraction**: Parses 360-element vectors to extract relevant features
4. **Frame Mapping**: Maps feature vector indices to original video frame indices

## 📊 Validation Benefits

### Direct Feature Validation
- Validates that velocity/acceleration calculations are correct
- Confirms that feature engineering produces expected values
- Shows real-time feature values overlaid on actual player movements

### Error Detection
- Identifies issues in feature engineering pipeline
- Catches synchronization problems between data sources
- Detects missing or incorrect feature calculations

### Performance Monitoring
- Provides visual confirmation of feature quality
- Enables spot-checking of engineered features
- Facilitates debugging of feature calculation issues