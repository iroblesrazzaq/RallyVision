#!/usr/bin/env python3
"""
Tennis Feature Engineer

This module contains the TennisFeatureEngineer class that handles
feature vector creation from preprocessed tennis pose data.
"""

import numpy as np
import os
from data_processor import DataProcessor

class TennisFeatureEngineer:
    """
    Creates feature vectors from preprocessed tennis pose data.
    
    This class takes preprocessed data with player assignments and creates
    feature vectors for training the tennis point detection model.
    """
    
    def __init__(self, screen_width=1280, screen_height=720, feature_vector_size=288):
        """
        Initialize the feature engineer.
        
        Args:
            screen_width (int): Width of the video frames
            screen_height (int): Height of the video frames
            feature_vector_size (int): Size of feature vectors (default: 288)
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.feature_vector_size = feature_vector_size
        self.data_processor = DataProcessor(screen_width, screen_height)
    
    def create_feature_vector(self, assigned_players, previous_assigned_players=None):
        """
        Create a feature vector from assigned players.
        
        Args:
            assigned_players (dict): Players with 'near_player' and 'far_player' keys
            previous_assigned_players (dict): Previous frame players for velocity calculations
            
        Returns:
            np.ndarray: Feature vector of size feature_vector_size
        """
        return self.data_processor.create_feature_vector(
            assigned_players, previous_assigned_players
        )
    
    def create_features_from_preprocessed(self, input_npz_path, output_file, overwrite=False):
        """
        Create feature vectors from preprocessed pose data.
        
        Args:
            input_npz_path (str): Path to preprocessed .npz file
            output_file (str): Path to output .npz file
            overwrite (bool): Whether to overwrite existing files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if output file already exists
            if os.path.exists(output_file) and not overwrite:
                print(f"  ✓ Already exists, skipping: {os.path.basename(output_file)}")
                return True
            
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
                feature_vector = self.create_feature_vector(assigned_players, previous_players)
                
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
                feature_array = np.empty((0, self.feature_vector_size))
                target_array = np.empty((0,))
                print(f"  No annotated frames found, creating empty arrays")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save feature vectors and targets
            np.savez_compressed(
                output_file,
                features=feature_array,
                targets=target_array
            )
            
            print(f"  ✓ Features saved to: {output_file}")
            return True
            
        except Exception as e:
            print(f"  ❌ Error processing {input_npz_path}: {e}")
            import traceback
            traceback.print_exc()
            return False

# Example usage
if __name__ == "__main__":
    # This would typically be in a separate script, but included for demonstration
    print("TennisFeatureEngineer - Example Usage")
    print("=" * 50)
    print("This class should be used from a separate processing script.")
    print("See create_features_pipeline.py for usage example.")