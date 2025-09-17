# Tennis Point Detector MVP

This tool automates the process of finding and extracting points from a raw tennis match video.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Add Model Assets:**
    Place your trained model files (`best_model.pth`, `scaler.joblib`, and `yolov8n-pose.pt`) into the `models/` directory. See `models/README.md` for details.
    
## Usage

Run the main script from the command line, providing a path to your video and your trained models.

```bash
python detect_points.py \
  --video /path/to/your/match.mp4 \
  --model models/best_model.pth \
  --scaler models/scaler.joblib \
  --output-dir output
```

## Outputs

The script will generate two files in the specified output directory:

- `match_segments.csv`: A CSV file with the start and end times of each detected point.
- `match_segmented.mp4`: A new video file containing only the clips of the detected points.
