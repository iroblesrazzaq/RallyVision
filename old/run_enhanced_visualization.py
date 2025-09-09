#!/usr/bin/env python3
"""
Utility script to run enhanced feature visualization for tennis pose data.
"""

import os
import sys
import glob
from data_scripts.enhanced_feature_visualizer import EnhancedFeatureVisualizer

def find_matching_files(video_name_pattern=None):
    """
    Find matching preprocessed, features, and video files.
    
    Args:
        video_name_pattern (str): Pattern to match video names (optional)
        
    Returns:
        list: List of tuples (preprocessed_path, features_path, video_path, base_name)
    """
    matches = []
    
    # Look for preprocessed files
    preprocessed_pattern = "pose_data/preprocessed/yolos_0.25conf_15fps_0s_to_99999s/*_preprocessed.npz"
    preprocessed_files = glob.glob(preprocessed_pattern)
    
    for preprocessed_path in preprocessed_files:
        # Extract base name
        base_name = os.path.basename(preprocessed_path).replace('_preprocessed.npz', '')
        
        # Skip if pattern specified and doesn't match
        if video_name_pattern and video_name_pattern.lower() not in base_name.lower():
            continue
            
        # Look for corresponding features file
        features_path = f"pose_data/features/yolos_0.25conf_15fps_0s_to_99999s/{base_name}_features.npz"
        
        # Look for corresponding video file (try different extensions)
        video_extensions = ['.mp4', '.mov', '.avi']
        video_path = None
        for ext in video_extensions:
            candidate_path = f"raw_videos/{base_name}{ext}"
            if os.path.exists(candidate_path):
                video_path = candidate_path
                break
        
        if os.path.exists(features_path) and video_path:
            matches.append((preprocessed_path, features_path, video_path, base_name))
    
    return matches

def run_visualization(preprocessed_path, features_path, video_path, base_name, 
                     start_time=0, duration=30, output_dir="sanity_check_clips"):
    """
    Run visualization for a specific set of files.
    
    Args:
        preprocessed_path (str): Path to preprocessed NPZ file
        features_path (str): Path to features NPZ file
        video_path (str): Path to video file
        base_name (str): Base name for output file
        start_time (int): Start time in seconds
        duration (int): Duration in seconds
        output_dir (str): Output directory
        
    Returns:
        bool: True if successful
    """
    # Create output path
    output_filename = f"enhanced_viz_{base_name}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = EnhancedFeatureVisualizer()
    
    print(f"Running visualization for: {base_name}")
    print(f"  Preprocessed: {preprocessed_path}")
    print(f"  Features: {features_path}")
    print(f"  Video: {video_path}")
    print(f"  Output: {output_path}")
    print(f"  Time: {start_time}s to {start_time + duration}s")
    
    # Run visualization
    success = visualizer.validate_and_visualize_features(
        preprocessed_path, features_path, video_path, output_path,
        start_time=start_time, duration=duration
    )
    
    if success:
        print(f"✅ Visualization completed successfully")
    else:
        print(f"❌ Visualization failed")
    
    return success

def main():
    """Main function to run visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Feature Visualizer for Tennis Pose Data")
    parser.add_argument("--video-pattern", "-p", help="Pattern to match video names")
    parser.add_argument("--start-time", "-s", type=int, default=0, help="Start time in seconds")
    parser.add_argument("--duration", "-d", type=int, default=30, help="Duration in seconds")
    parser.add_argument("--output-dir", "-o", default="sanity_check_clips", help="Output directory")
    parser.add_argument("--list", "-l", action="store_true", help="List available videos")
    parser.add_argument("--all", "-a", action="store_true", help="Process all available videos")
    
    args = parser.parse_args()
    
    # Find matching files
    matches = find_matching_files(args.video_pattern)
    
    if args.list:
        print("Available videos:")
        for i, (preprocessed_path, features_path, video_path, base_name) in enumerate(matches):
            print(f"  {i+1}. {base_name}")
        return
    
    if not matches:
        print("No matching files found.")
        return
    
    print(f"Found {len(matches)} matching video sets")
    
    if args.all:
        # Process all videos
        for preprocessed_path, features_path, video_path, base_name in matches:
            run_visualization(preprocessed_path, features_path, video_path, base_name,
                            args.start_time, args.duration, args.output_dir)
    else:
        # Process first matching video
        preprocessed_path, features_path, video_path, base_name = matches[0]
        run_visualization(preprocessed_path, features_path, video_path, base_name,
                         args.start_time, args.duration, args.output_dir)

if __name__ == "__main__":
    main()