# Tennis Point Detection Dataset Creator

This tool creates train/validation/test datasets from feature-engineered tennis data for LSTM model training.

## 🎯 Purpose

The dataset creator processes feature-engineered tennis data to:
1. Prevent data leakage through temporal splitting
2. Generate overlapping sequences for training
3. Save data in an efficient format for model training

## 📁 Data Structure

### Input
```
pose_data/features/yolos_0.25conf_15fps_0s_to_99999s/
├── video1_features.npz  # (N, 360) features, (N,) targets
├── video2_features.npz
└── ...
```

### Output
```
processed_data/
├── train.h5        # Training sequences from all videos
├── val.h5          # Validation sequences from all videos
├── test.h5         # Test sequences from all videos
└── dataset_info.json  # Metadata about dataset creation
```

## 🚀 Usage

### Basic Usage
```python
from data_scripts.dataset_creator import TennisDatasetCreator

# Initialize creator
creator = TennisDatasetCreator(
    feature_dir="pose_data/features/yolos_0.25conf_15fps_0s_to_99999s",
    output_dir="processed_data",
    splits=(0.765, 0.085, 0.15)  # train/val/test ratios
)

# Create dataset
stats = creator.create_dataset()
```

## 📊 Dataset Creation Process

### 1. Temporal Splitting
Each video is split temporally to prevent data leakage:
- **Train**: First 76.5% of frames
- **Validation**: Next 8.5% of frames  
- **Test**: Last 15% of frames

### 2. Sequence Generation
Overlapping sequences are created from each split:
- **Sequence length**: 150 frames (10 seconds at 15 FPS)
- **Overlap**: 75 frames (50% overlap)
- **Example**: 300 frames → 3 sequences: [0-149], [75-224], [150-299]

### 3. Data Storage
Data is stored in HDF5 format for efficient loading:
- **Compression**: gzip compression reduces file size
- **Partial loading**: Data loader can load batches without loading entire dataset
- **Metadata**: dataset_info.json contains creation statistics

## ⚙️ Parameters

### Split Ratios
- `train_ratio`: 0.765 (76.5% of video for training)
- `val_ratio`: 0.085 (8.5% of video for validation)
- `test_ratio`: 0.15 (15% of video for testing)

### Sequence Parameters
- `sequence_length`: 150 frames (10 seconds)
- `overlap`: 75 frames (50% overlap)
- `num_features`: 360 features per frame

## 📈 Data Flow

```
Raw Videos (25-30 FPS)
        ↓
Pose Detection & Feature Engineering
        ↓
Feature NPZ Files (15 FPS, 360 features/frame)
        ↓
TennisDatasetCreator
        ↓
Temporal Splits (per video)
        ↓
Sequence Generation (150-frame overlapping sequences)
        ↓
HDF5 Files (train.h5, val.h5, test.h5)
```

## 🎯 Benefits

### Data Leakage Prevention
- Temporal splitting ensures no future information leaks into training
- Each video's splits are processed independently
- Maintains temporal integrity within each split

### Efficient Processing
- HDF5 format enables efficient partial loading
- Overlapping sequences maximize data usage
- Cross-video combination improves generalization

### Flexible Usage
- Configurable split ratios
- Standardized output format
- Metadata for experiment tracking