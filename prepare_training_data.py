#!/usr/bin/env python3
"""
Script to convert processed pose data to LSTM-ready feature vectors.

This script processes the filtered pose data (.npz files) and converts them 
to feature vectors that can be used for training the LSTM model.
"""

import os
import numpy as np
import argparse
import glob
from data_processor import DataProcessor

def process_pose_data_to_features(npz_file_path, output_dir=None):
    """
    Process a single pose data file and convert it to feature vectors.
    
    Args:
        npz_file_path (str): Path to the .npz file containing pose data
        output_dir (str): Directory to save the feature vectors. If None, saves in the same directory as the input file.
        
    Returns:
        str: Path to the saved feature vector file
    """
    print(f"Processing {npz_file_path}")
    
    # Load pose data
    try:
        pose_data = np.load(npz_file_path, allow_pickle=True)['frames']
        print(f"Loaded {len(pose_data)} frames")
    except Exception as e:
        print(f"Error loading {npz_file_path}: {e}")
        return None
    
    # Initialize data processor
    processor = DataProcessor(screen_width=1280, screen_height=720)
    
    # Process each frame to create feature vectors
    feature_vectors = []
    annotation_status = []  # Track annotation status for each frame
    previous_players = None
    
    for frame_idx, frame_data in enumerate(pose_data):
        # Check annotation status
        status = frame_data.get('annotation_status', 0)  # Default to 0 if not present
        annotation_status.append(status)
        
        # Skip frames that were not annotated (-100)
        if status < 0:
            continue
            
        # Assign players using the core heuristic
        assigned_players = processor.assign_players(frame_data)
        
        # Convert the assignment into a feature vector
        feature_vector = processor.create_feature_vector(assigned_players, previous_players)
        
        feature_vectors.append(feature_vector)
        previous_players = assigned_players
        
        # Progress indicator
        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1}/{len(pose_data)} frames")
    
    # Convert to numpy arrays
    feature_array = np.array(feature_vectors)
    status_array = np.array(annotation_status)
    
    print(f"Feature array shape: {feature_array.shape}")
    print(f"Annotation status array shape: {status_array.shape}")
    print(f"Frames with status >= 0: {np.sum(status_array >= 0)}")
    print(f"Frames with status == -100 (skipped): {np.sum(status_array == -100)}")
    
    # Determine output path
    if output_dir is None:
        output_dir = os.path.dirname(npz_file_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save feature vectors
    base_name = os.path.splitext(os.path.basename(npz_file_path))[0]
    feature_file_path = os.path.join(output_dir, f"{base_name}_features.npy")
    status_file_path = os.path.join(output_dir, f"{base_name}_status.npy")
    
    try:
        np.save(feature_file_path, feature_array)
        np.save(status_file_path, status_array)
        print(f"Saved features to {feature_file_path}")
        print(f"Saved annotation status to {status_file_path}")
        return feature_file_path
    except Exception as e:
        print(f"Error saving features to {feature_file_path}: {e}")
        return None

def process_all_pose_data(input_dir, output_dir=None):
    """
    Process all pose data files in a directory.
    
    Args:
        input_dir (str): Directory containing .npz files
        output_dir (str): Directory to save feature vectors. If None, saves in subdirectories of input_dir.
    """
    # Find all .npz files in the input directory
    npz_files = glob.glob(os.path.join(input_dir, "*.npz"))
    
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return
    
    print(f"Found {len(npz_files)} .npz files to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for npz_file in npz_files:
        try:
            result = process_pose_data_to_features(npz_file, output_dir)
            if result:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Failed to process {npz_file}: {e}")
            failed += 1
    
    print(f"\nProcessing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

def main():
    parser = argparse.ArgumentParser(description='Convert pose data to LSTM-ready feature vectors')
    parser.add_argument('--input', type=str, required=True,
                        help='Input .npz file or directory containing .npz files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for feature vectors (default: same as input)')
    
    args = parser.parse_args()
    
    # Check if input is a file or directory
    if os.path.isfile(args.input) and args.input.endswith('.npz'):
        # Process single file
        process_pose_data_to_features(args.input, args.output)
    elif os.path.isdir(args.input):
        # Process all files in directory
        process_all_pose_data(args.input, args.output)
    else:
        print(f"Error: {args.input} is not a valid .npz file or directory")

if __name__ == "__main__":
    main()