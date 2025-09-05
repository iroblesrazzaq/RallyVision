#!/usr/bin/env python3
"""
Test script for TennisDataset class.
"""

import numpy as np
import os
import tempfile
import pandas as pd
from tennis_dataset import TennisDataset, create_data_loaders


def create_test_data():
    """Create temporary test data for testing the dataset."""
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, "features")
    annotations_dir = os.path.join(temp_dir, "annotations")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Create test feature vectors (300 frames, 288 features)
    test_features = np.random.randn(300, 288).astype(np.float32)
    
    # Add some missing data (-1 values)
    test_features[50:60, :] = -1.0
    test_features[100:110, :] = -1.0
    
    # Save feature file
    feature_file = os.path.join(data_dir, "test_video_features.npy")
    np.save(feature_file, test_features)
    
    # Create test annotations
    annotations_data = {
        'start_frame': [20, 150, 250],
        'end_frame': [40, 170, 270],
        'label': ['point', 'point', 'point']
    }
    annotations_df = pd.DataFrame(annotations_data)
    
    # Save annotation file
    annotation_file = os.path.join(annotations_dir, "test_video_annotations.csv")
    annotations_df.to_csv(annotation_file, index=False)
    
    return temp_dir, data_dir, annotations_dir


def test_tennis_dataset():
    """Test the TennisDataset class."""
    print("Testing TennisDataset...")
    
    # Create test data
    temp_dir, data_dir, annotations_dir = create_test_data()
    
    try:
        # Initialize dataset
        dataset = TennisDataset(
            data_dir=data_dir,
            annotations_dir=annotations_dir,
            sequence_length=150,
            feature_vector_size=288,
            normalize=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test getting a sample
            sequence, label = dataset[0]
            print(f"Sample sequence shape: {sequence.shape}")
            print(f"Sample label shape: {label.shape}")
            print(f"Sample label value: {label.item()}")
            
            # Test a few more samples
            for i in range(min(3, len(dataset))):
                seq, lbl = dataset[i]
                print(f"Sample {i}: sequence {seq.shape}, label {lbl.item()}")
        
        # Test data loaders
        print("\nTesting data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=data_dir,
            annotations_dir=annotations_dir,
            batch_size=2,
            sequence_length=150,
            feature_vector_size=288,
            train_split=0.6,
            val_split=0.2,
            test_split=0.2
        )
        
        # Test iterating through train loader
        print("Train loader batches:")
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: sequences {sequences.shape}, labels {labels.shape}")
            if batch_idx >= 2:  # Only test first 3 batches
                break
                
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup temporary files
        import shutil
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    test_tennis_dataset()