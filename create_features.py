#!/usr/bin/env python3
"""
Feature Engineering Script for Tennis Point Detection

This script takes preprocessed pose data and creates feature vectors for training.
"""

import os
import sys
import argparse
import numpy as np
import time
import glob
from data_processor import DataProcessor

def create_features_from_preprocessed(input_npz_path, output_dir, overwrite=False):
    """
    Create feature vectors from preprocessed pose data.
    
    Args:
        input_npz_path (str): Path to preprocessed .npz file
        output_dir (str): Directory to save feature vectors
        overwrite (bool): Whether to overwrite existing files
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load preprocessed data
        print(f"  Loading preprocessed data from: {input_npz_path}")
        data = np.load(input_npz_path, allow_pickle=True)
        
        # Extract arrays
        frames = data['frames']
        targets = data['targets']
        near_players = data['near_players']
        far_players = data['far_players']
        
        print(f"  Loaded {len(frames)} frames")
        print(f"  Annotation status distribution:")
        print(f"    -100 (skipped): {np.sum(targets == -100)}")
        print(f"    0 (not in play): {np.sum(targets == 0)}")
        print(f"    1 (in play): {np.sum(targets == 1)}")
        
        # Initialize data processor
        processor = DataProcessor(screen_width=1280, screen_height=720)
        
        # Create feature vectors only for annotated frames (status >= 0)
        annotated_indices = np.where(targets >= 0)[0]
        print(f"  Creating features for {len(annotated_indices)} annotated frames")
        
        feature_vectors = []
        feature_targets = []
        previous_players = None
        
        for idx in annotated_indices:
            # Create assigned players dictionary for this frame
            assigned_players = {
                'near_player': near_players[idx],
                'far_player': far_players[idx]
            }
            
            # Create feature vector
            feature_vector = processor.create_feature_vector(assigned_players, previous_players)
            
            feature_vectors.append(feature_vector)
            feature_targets.append(targets[idx])
            
            # Update previous players for velocity/acceleration calculations
            previous_players = assigned_players
            
            # Progress indicator
            if len(feature_vectors) % 100 == 0:
                print(f"    Created features for {len(feature_vectors)}/{len(annotated_indices)} annotated frames")
        
        # Convert to numpy arrays
        if feature_vectors:
            feature_array = np.array(feature_vectors)
            target_array = np.array(feature_targets)
            
            print(f"  Feature array shape: {feature_array.shape}")
            print(f"  Target array shape: {target_array.shape}")
        else:
            # Create empty arrays with correct shapes
            feature_array = np.empty((0, 288))  # 288 is the expected feature vector size
            target_array = np.empty((0,))
            print(f"  No annotated frames found, creating empty arrays")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create descriptive filename based on input file
        base_name = os.path.splitext(os.path.basename(input_npz_path))[0]
        base_name = base_name.replace("_preprocessed", "")  # Remove _preprocessed suffix
        
        # Save feature vectors and targets
        feature_file = os.path.join(output_dir, f"{base_name}_features.npz")
        
        # Check if output file already exists
        if os.path.exists(feature_file) and not overwrite:
            print(f"  ✓ Already exists, skipping: {os.path.basename(feature_file)}")
            return True
        
        np.savez_compressed(
            feature_file,
            features=feature_array,
            targets=target_array
        )
        
        print(f"  ✓ Features saved to: {feature_file}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {input_npz_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_all_preprocessed(input_dir, output_dir, overwrite=False):
    """
    Process all preprocessed files in a directory.
    
    Args:
        input_dir (str): Directory containing preprocessed .npz files
        output_dir (str): Directory to save feature vectors
        overwrite (bool): Whether to overwrite existing files
    """
    # Find all preprocessed .npz files in the input directory
    npz_files = glob.glob(os.path.join(input_dir, "*_preprocessed.npz"))
    
    if not npz_files:
        print(f"❌ No preprocessed .npz files found in {input_dir}")
        return
    
    print(f"Found {len(npz_files)} preprocessed files to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for npz_file in npz_files:
        try:
            print(f"Processing: {os.path.basename(npz_file)}")
            if create_features_from_preprocessed(npz_file, output_dir, overwrite):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Failed to process {npz_file}: {e}")
            failed += 1
    
    print(f"\nFeature engineering complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Create feature vectors from preprocessed tennis pose data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python create_features.py --input-dir "preprocessed_data" --output-dir "training_features"
    
    python create_features.py --input-dir "preprocessed_data" --output-dir "training_features" --overwrite
        """
    )
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing preprocessed .npz files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save feature vectors')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing feature files')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory {args.input_dir} does not exist")
        return
    
    print("=== Tennis Feature Engineering ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Overwrite: {'Yes' if args.overwrite else 'No'}")
    print()
    
    # Start timing
    start_time = time.time()
    
    # Process all preprocessed files
    process_all_preprocessed(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite
    )
    
    # Calculate and print total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    
    print(f"\n🎯 Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")

if __name__ == "__main__":
    main()