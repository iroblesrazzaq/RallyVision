#!/usr/bin/env python3
"""
Tennis Data Preprocessor

This module contains the TennisDataPreprocessor class that handles
court filtering and player assignment in a single, cohesive class.
"""

import numpy as np
import os
from court_detector import CourtDetector
from data_processor import DataProcessor

class TennisDataPreprocessor:
    """
    Preprocesses tennis pose data by applying court filtering and player assignment.
    
    This class combines court mask filtering with player assignment to create
    a preprocessed dataset that can be used for visualization and feature engineering.
    """
    
    def __init__(self, screen_width=1280, screen_height=720):
        """
        Initialize the preprocessor.
        
        Args:
            screen_width (int): Width of the video frames
            screen_height (int): Height of the video frames
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.data_processor = DataProcessor(screen_width, screen_height)
    
    def generate_court_mask(self, video_path):
        """
        Generate a court mask from a video file.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            np.ndarray or None: Court mask or None if detection failed
        """
        try:
            detector = CourtDetector()
            mask, clean_frame, metadata = detector.process_video(video_path, target_time=60)
            return mask
        except Exception as e:
            print(f"  ⚠️  Error during court detection: {e}")
            return None
    
    def filter_frame_by_court(self, frame_data, mask):
        """
        Filter a single frame's detections by court mask.
        
        Args:
            frame_data (dict): Frame data with 'boxes', 'keypoints', 'conf'
            mask (np.ndarray): Court mask
            
        Returns:
            dict: Filtered frame data
        """
        if mask is None:
            # No filtering applied
            return frame_data
        
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
        return {
            'boxes': np.array(kept_boxes),
            'keypoints': np.array(kept_keypoints),
            'conf': np.array(kept_conf)
        }
    
    def assign_players_to_frame(self, frame_data):
        """
        Assign near and far players to a frame.
        
        Args:
            frame_data (dict): Frame data with 'boxes', 'keypoints', 'conf'
            
        Returns:
            dict: Assigned players with 'near_player' and 'far_player' keys
        """
        return self.data_processor.assign_players(frame_data)
    
    def preprocess_single_video(self, input_npz_path, video_path, output_npz_path, overwrite=False):
        """
        Preprocess a single video's pose data.
        
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
            mask = self.generate_court_mask(video_path)
            if mask is not None:
                print(f"  ✓ Generated mask: {mask.shape}")
            else:
                print(f"  ⚠️  No court mask available - processing without filtering")
            
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
                
                # Apply court filtering
                filtered_frame_data = self.filter_frame_by_court(frame_data, mask)
                
                # Store filtered frame data
                all_frame_data.append(filtered_frame_data)
                
                # Apply player assignment
                assigned_players = self.assign_players_to_frame(filtered_frame_data)
                
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

# Example usage
if __name__ == "__main__":
    # This would typically be in a separate script, but included for demonstration
    print("TennisDataPreprocessor - Example Usage")
    print("=" * 50)
    print("This class should be used from a separate processing script.")
    print("See preprocess_data_pipeline.py for usage example.")