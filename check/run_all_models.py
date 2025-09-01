#!/usr/bin/env python3
"""
Run all YOLO model sizes on videos for comparison
"""

import subprocess
import sys
import os
from pathlib import Path

def run_model_on_videos(model_size, num_videos=3):
    """Run a specific model size on a subset of videos"""
    print(f"\n{'='*50}")
    print(f"Running YOLO {model_size.upper()} model on {num_videos} videos")
    print(f"{'='*50}")
    
    # Create output directory for this model
    output_dir = f"./annotated_videos_{model_size}"
    
    cmd = [
        "python", "yolo_player_detector.py",
        "--model-size", model_size,
        "--output-dir", output_dir,
        "--duration", "2"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print(f"✓ {model_size.upper()} model completed successfully")
            print(f"Output saved to: {output_dir}")
        else:
            print(f"✗ {model_size.upper()} model failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Error running {model_size.upper()} model: {e}")
        return False
    
    return True

def main():
    """Run all model sizes"""
    print("YOLO Model Comparison - Tennis Player Detection")
    print("This will run all model sizes on your videos for comparison")
    
    # Check if main script exists
    if not os.path.exists("yolo_player_detector.py"):
        print("Error: yolo_player_detector.py not found in current directory")
        sys.exit(1)
    
    # Model sizes to test
    models = ['n', 's', 'm', 'l']
    
    print(f"\nWill test models: {', '.join(models)}")
    print("Note: This may take a while, especially for larger models")
    
    # Ask for confirmation
    response = input("\nContinue? (y/N): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return
    
    # Run each model
    successful_models = []
    for model in models:
        if run_model_on_videos(model):
            successful_models.append(model)
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Successful models: {', '.join(successful_models)}")
    
    if successful_models:
        print(f"\nAnnotated videos saved to:")
        for model in successful_models:
            output_dir = f"./annotated_videos_{model}"
            print(f"  {model.upper()}: {output_dir}")
        
        print(f"\nYou can now compare the results between different model sizes!")
        print("Larger models generally provide better accuracy but slower processing.")
    else:
        print("No models completed successfully. Check the errors above.")

if __name__ == "__main__":
    main()
