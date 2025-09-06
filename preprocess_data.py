#!/usr/bin/env python3
"""
Data Preprocessing Script for Tennis Point Detection

This script combines court mask filtering with player assignment to create
a preprocessed dataset that can be used for visualization and feature engineering.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import time
import glob
from data_processor import DataProcessor
from court_detector import CourtDetector

def preprocess_single_video(input_npz_path, video_path, output_npz_path, overwrite=False):
    """
    Preprocess a single video's pose data by applying court filtering and player assignment.
    
    Args:
        input_npz_path (str): Path to input .npz file with raw pose data
        video_path (str): Path to the video file for mask generation
        output_npz_path (str): Path to output .npz file
        overwrite (bool): Whether to overwrite existing files
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if output file already exists
        if os.path.exists(output_npz_path) and not overwrite:
            print(f"  ✓ Already exists, skipping: {os.path.basename(output_npz_path)}")
            return True
            
        # Load pose data
        print(f"  Loading pose data from: {input_npz_path}")
        pose_data = np.load(input_npz_path, allow_pickle=True)['frames']
        print(f"  Loaded {len(pose_data)} frames")
        
        # Generate court mask
        print(f"  Generating court mask from: {video_path}")
        try:
            detector = CourtDetector()
            mask, clean_frame, metadata = detector.process_video(video_path, target_time=60)
            
            if mask is None:
                print(f"  ⚠️  Court detection failed: {metadata.get('error', 'Unknown error')}")
                mask = None
            else:
                print(f"  ✓ Generated mask: {mask.shape}")
        except Exception as e:
            print(f"  ⚠️  Error during court detection: {e}")
            mask = None
        
        # Initialize data processor
        processor = DataProcessor(screen_width=1280, screen_height=720)
        
        # Initialize data structures for results
        all_frame_data = []
        all_targets = []
        all_near_players = []
        all_far_players = []
        
        # Process each frame
        print("  Processing frames...")
        for frame_idx, frame_data in enumerate(pose_data):
            # Get annotation status
            annotation_status = frame_data.get('annotation_status', 0)
            all_targets.append(annotation_status)
            
            # Skip frames that weren't annotated
            if annotation_status == -100:
                # Store empty data for unannotated frames
                all_frame_data.append({
                    'boxes': np.array([]),
                    'keypoints': np.array([]),
                    'conf': np.array([])
                })
                all_near_players.append(None)
                all_far_players.append(None)
                continue
            
            # Apply court filtering if mask is available
            if mask is not None:
                # Extract arrays from frame data
                boxes = frame_data['boxes']
                keypoints = frame_data['keypoints']
                conf = frame_data['conf']
                
                # Initialize empty lists for the current frame's surviving data
                kept_boxes = []
                kept_keypoints = []
                kept_conf = []
                
                # Filter each person in the frame
                for i, box in enumerate(boxes):
                    # Calculate bounding box centroid
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    
                    # Check if centroid is inside the playable area
                    # First, perform boundary check to prevent IndexError
                    if (0 <= center_y < mask.shape[0] and 
                        0 <= center_x < mask.shape[1] and 
                        mask[int(center_y), int(center_x)] == 0):
                        
                        # Keep this person's data
                        kept_boxes.append(box)
                        kept_keypoints.append(keypoints[i])
                        kept_conf.append(conf[i])
                
                # Assemble filtered frame data
                filtered_frame_data = {
                    'boxes': np.array(kept_boxes),
                    'keypoints': np.array(kept_keypoints),
                    'conf': np.array(kept_conf)
                }
            else:
                # No filtering applied
                filtered_frame_data = frame_data
            
            # Store filtered frame data
            all_frame_data.append(filtered_frame_data)
            
            # Apply player assignment
            assigned_players = processor.assign_players(filtered_frame_data)
            
            # Store player assignments
            all_near_players.append(assigned_players['near_player'])
            all_far_players.append(assigned_players['far_player'])
            
            # Progress indicator
            if (frame_idx + 1) % 100 == 0:
                print(f"    Processed {frame_idx + 1}/{len(pose_data)} frames")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_npz_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the preprocessed data
        np.savez_compressed(
            output_npz_path,
            frames=all_frame_data,
            targets=np.array(all_targets),
            near_players=all_near_players,
            far_players=all_far_players
        )
        
        print(f"  ✓ Preprocessed data saved to: {output_npz_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {input_npz_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def preprocess_all_videos(input_dir, video_dir, output_dir, overwrite=False):
    """
    Preprocess all videos in a directory.
    
    Args:
        input_dir (str): Directory containing .npz files
        video_dir (str): Directory containing video files
        output_dir (str): Directory to save preprocessed .npz files
        overwrite (bool): Whether to overwrite existing files
    """
    # Find all .npz files in the input directory
    npz_files = glob.glob(os.path.join(input_dir, "*.npz"))
    
    if not npz_files:
        print(f"❌ No .npz files found in {input_dir}")
        return
    
    print(f"Found {len(npz_files)} .npz files to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for npz_file in npz_files:
        try:
            # Derive video name from npz file name
            base_name = os.path.splitext(os.path.basename(npz_file))[0]
            # Remove the suffix to get the video name
            # e.g., "video_name_posedata_0s_to_60s_yolos" -> "video_name"
            video_name = "_".join(base_name.split("_posedata")[0].split("_")[:-1]) if "_posedata" in base_name else base_name.split("_posedata")[0]
            
            # Look for corresponding video file
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
            video_file = None
            
            for ext in video_extensions:
                pattern = os.path.join(video_dir, f"{video_name}.{ext.replace('*', '')}")
                if os.path.exists(pattern):
                    video_file = pattern
                    break
                # Also try without extension matching
                pattern = os.path.join(video_dir, f"{video_name}{ext[1:]}")
                if os.path.exists(pattern):
                    video_file = pattern
                    break
            
            if video_file is None:
                # Try a more general pattern
                for ext in video_extensions:
                    pattern = os.path.join(video_dir, ext)
                    matching_files = glob.glob(pattern)
                    for match in matching_files:
                        if video_name in os.path.basename(match):
                            video_file = match
                            break
                    if video_file:
                        break
            
            if video_file is None:
                print(f"  ⚠️  No matching video file found for {base_name}, skipping...")
                failed += 1
                continue
            
            # Create output file path
            output_file = os.path.join(output_dir, f"{base_name}_preprocessed.npz")
            
            print(f"Processing: {os.path.basename(npz_file)}")
            if preprocess_single_video(npz_file, video_file, output_file, overwrite):
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"Failed to process {npz_file}: {e}")
            failed += 1
    
    print(f"\nPreprocessing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Preprocess tennis pose data for point detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python preprocess_data.py --input-dir "pose_data/unfiltered/yolos_0.05conf_15fps_0s_to_60s" --video-dir "raw_videos" --output-dir "preprocessed_data"
    
    python preprocess_data.py --input-dir "pose_data/unfiltered/yolos_0.05conf_15fps_0s_to_60s" --video-dir "raw_videos" --output-dir "preprocessed_data" --overwrite
        """
    )
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing raw pose .npz files')
    parser.add_argument('--video-dir', type=str, default='raw_videos',
                        help='Directory containing video files (default: raw_videos)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save preprocessed .npz files')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing preprocessed files')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory {args.input_dir} does not exist")
        return
    
    # Check if video directory exists
    if not os.path.exists(args.video_dir):
        print(f"❌ Error: Video directory {args.video_dir} does not exist")
        return
    
    print("=== Tennis Data Preprocessing ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Video directory: {args.video_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Overwrite: {'Yes' if args.overwrite else 'No'}")
    print()
    
    # Start timing
    start_time = time.time()
    
    # Preprocess all videos
    preprocess_all_videos(
        input_dir=args.input_dir,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite
    )
    
    # Calculate and print total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    
    print(f"\n🎯 Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")

if __name__ == "__main__":
    main()