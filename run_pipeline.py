#!/usr/bin/env python3
"""
Pipeline Runner Script

This script demonstrates the complete refactored pipeline workflow.
"""

import os
import argparse
import subprocess
import sys

def run_command(command, description):
    """Run a command and print its output."""
    print(f"\n=== {description} ===")
    print(f"Running: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run the complete refactored tennis point detection pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Stages:
    1. Original YOLO Inference (assumed already done)
    2. Data Preprocessing
    3. Feature Engineering

Example workflow:
    python run_pipeline.py --input-data "pose_data/unfiltered/yolos_0.05conf_15fps_15s_to_60s" --video-dir "raw_videos" --output-dir "processed_data"
        """
    )
    
    parser.add_argument('--input-data', type=str, required=True,
                        help='Directory containing raw pose .npz files')
    parser.add_argument('--video-dir', type=str, default='raw_videos',
                        help='Directory containing video files (default: raw_videos)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Base directory for all output')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files')
    
    args = parser.parse_args()
    
    # Create output directories
    preprocessed_dir = os.path.join(args.output_dir, "preprocessed")
    features_dir = os.path.join(args.output_dir, "features")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    
    print("=== Tennis Point Detection Pipeline Runner ===")
    print(f"Input data: {args.input_data}")
    print(f"Video directory: {args.video_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Overwrite: {'Yes' if args.overwrite else 'No'}")
    print()
    
    # Stage 2: Data Preprocessing
    preprocess_cmd = (
        f"python preprocess_data.py "
        f"--input-dir '{args.input_data}' "
        f"--video-dir '{args.video_dir}' "
        f"--output-dir '{preprocessed_dir}'"
    )
    
    if args.overwrite:
        preprocess_cmd += " --overwrite"
    
    if not run_command(preprocess_cmd, "Stage 2: Data Preprocessing"):
        print("Preprocessing failed. Exiting.")
        return False
    
    # Stage 3: Feature Engineering
    features_cmd = (
        f"python create_features.py "
        f"--input-dir '{preprocessed_dir}' "
        f"--output-dir '{features_dir}'"
    )
    
    if args.overwrite:
        features_cmd += " --overwrite"
    
    if not run_command(features_cmd, "Stage 3: Feature Engineering"):
        print("Feature engineering failed. Exiting.")
        return False
    
    print("\n=== Pipeline Complete ===")
    print("Preprocessed data saved to:", preprocessed_dir)
    print("Feature vectors saved to:", features_dir)
    print("\nThe data is now ready for model training!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)