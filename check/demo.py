#!/usr/bin/env python3
"""
Demo script showing YOLO Player Detector usage
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and show description"""
    print(f"\n{'='*60}")
    print(f"DEMO: {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()
    
    try:
        result = subprocess.run(cmd, shell=True, text=True)
        if result.returncode == 0:
            print(f"\n✓ Command completed successfully!")
        else:
            print(f"\n✗ Command failed with exit code {result.returncode}")
    except Exception as e:
        print(f"\n✗ Error running command: {e}")

def main():
    """Run demo commands"""
    print("YOLO Player Detector - Demo Script")
    print("This script demonstrates various usage patterns")
    
    # Check if we're in the right directory
    if not os.path.exists("yolo_player_detector.py"):
        print("Error: Please run this script from the 'check' directory")
        sys.exit(1)
    
    # Demo 1: Process single video with nano model
    demo1_cmd = 'python yolo_player_detector.py --single-video "../raw_videos/Aditi Narayan ｜ Matchplay.mp4" --model-size n --duration 1'
    run_command(demo1_cmd, "Single video with nano model (1 minute)")
    
    # Demo 2: Process single video with small model
    demo2_cmd = 'python yolo_player_detector.py --single-video "../raw_videos/Monica Greene unedited tennis match play.mp4" --model-size s --duration 1'
    run_command(demo2_cmd, "Single video with small model (1 minute)")
    
    # Demo 3: Process single video with medium model
    demo3_cmd = 'python yolo_player_detector.py --single-video "../raw_videos/Brady Knackstedt (Blue Shirt⧸Black Shorts)(4.0 UTR) Unedited Match Play vs. opponent (5.54 UTR).mp4" --model-size m --duration 1'
    run_command(demo3_cmd, "Single video with medium model (1 minute)")
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE!")
    print(f"{'='*60}")
    print("You can now:")
    print("1. Compare the annotated videos in the 'annotated_videos' directory")
    print("2. Run 'python yolo_player_detector.py' to process all videos")
    print("3. Use 'python run_all_models.py' to compare all model sizes")
    print("4. Check the README.md for more options")

if __name__ == "__main__":
    main()
