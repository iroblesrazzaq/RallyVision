# Progress Bar Handling

The tennis data pipeline now includes enhanced progress tracking with visual progress bars:

## Features

- **Visual Progress Bars**: Using tqdm for professional-looking progress indicators
- **Real-time Updates**: Progress updates in real-time during processing
- **Clean Output**: Single updating line instead of thousands of output lines
- **Frame Counter**: Displays processed frames vs total frames
- **ETA Display**: Shows estimated time remaining
- **Processing Rate**: Displays frames per second processing speed

## Implementation Details

- **Pose Extractor**: Uses tqdm progress bar that updates in place
- **Pipeline Runner**: Inherits stdout/stderr from subprocesses to show live progress
- **No Output Spam**: Progress bars update on the same line instead of creating new lines

## Usage

When running the pipeline, progress bars will be displayed directly from the subprocess:
```bash
python run_pipeline.py --config data_configs/config1.json
```

The progress bar will show:
- Percentage completion
- Visual progress bar
- Processed frames count
- Estimated time remaining
- Processing speed (it/s)