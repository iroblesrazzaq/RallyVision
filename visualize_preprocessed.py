#!/usr/bin/env python3
"""
Visualization Helper Script

This script demonstrates how preprocessed data could be visualized.
It's not a complete visualization tool, but shows the data structure
that would be used for visualization.
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

def visualize_preprocessed_data(npz_file, max_frames=10):
    """
    Visualize the structure of preprocessed data.
    
    Args:
        npz_file (str): Path to preprocessed .npz file
        max_frames (int): Maximum number of frames to show info for
    """
    print(f"Loading preprocessed data from: {npz_file}")
    
    # Load data
    data = np.load(npz_file, allow_pickle=True)
    
    frames = data['frames']
    targets = data['targets']
    near_players = data['near_players']
    far_players = data['far_players']
    
    print(f"\nData Structure:")
    print(f"  Total frames: {len(frames)}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Near players: {len(near_players)}")
    print(f"  Far players: {len(far_players)}")
    
    print(f"\nAnnotation Status Distribution:")
    print(f"  -100 (skipped): {np.sum(targets == -100)}")
    print(f"  0 (not in play): {np.sum(targets == 0)}")
    print(f"  1 (in play): {np.sum(targets == 1)}")
    
    print(f"\nFirst {min(max_frames, len(frames))} frames:")
    for i in range(min(max_frames, len(frames))):
        print(f"  Frame {i}:")
        print(f"    Target: {targets[i]}")
        print(f"    Near player: {'Present' if near_players[i] is not None else 'None'}")
        print(f"    Far player: {'Present' if far_players[i] is not None else 'None'}")
        
        if frames[i]['boxes'].size > 0:
            print(f"    Detections: {len(frames[i]['boxes'])}")
        else:
            print(f"    Detections: 0")
    
    # Show feature vector shape if features exist
    if 'features' in data:
        features = data['features']
        print(f"\nFeature data:")
        print(f"  Features shape: {features.shape}")
        if len(features) > 0:
            print(f"  First feature vector sample: {features[0][:10]}...")  # Show first 10 values

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Visualize preprocessed tennis pose data structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python visualize_preprocessed.py preprocessed_data/video_name_preprocessed.npz
        """
    )
    
    parser.add_argument('npz_file', help='Path to preprocessed .npz file')
    parser.add_argument('--max-frames', type=int, default=10,
                        help='Maximum number of frames to show info for (default: 10)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.npz_file):
        print(f"Error: File {args.npz_file} does not exist")
        return False
    
    visualize_preprocessed_data(args.npz_file, args.max_frames)
    return True

if __name__ == "__main__":
    main()