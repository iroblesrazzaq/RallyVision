"""
Enhanced Feature Visualizer for Tennis Pose Data

This module contains the EnhancedFeatureVisualizer class that validates
engineered features by mapping feature vectors to original video frames.
It combines preprocessed data (for court mask and frame timing) with
feature vectors (for engineered features like velocity and acceleration).
"""

import numpy as np
import cv2
import os
import glob
from pathlib import Path

# COCO keypoint connections
COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

# Colors for different elements
COLORS = {
    'bbox': (0, 255, 0),      # Green
    'keypoints': (0, 0, 255),  # Red
    'connections': (255, 0, 0),  # Blue
    'player1': (255, 165, 0),  # Orange
    'player2': (128, 0, 128),  # Purple
    'velocity': (0, 255, 255),  # Cyan for velocity vectors
    'acceleration': (255, 0, 255),  # Magenta for acceleration vectors
    'point_indicator': (0, 255, 255),  # Cyan for point indicator
    'not_point_indicator': (0, 0, 255),  # Red for not in point
    'skipped_indicator': (128, 128, 128),  # Gray for skipped frames
    'feature_text': (255, 255, 255)  # White for feature text
}

class EnhancedFeatureVisualizer:
    """
    Validates engineered features by mapping feature vectors to original video frames.
    Combines preprocessed data with feature vectors for comprehensive validation.
    """
    
    def __init__(self, screen_width=1280, screen_height=720):
        """
        Initialize the enhanced feature visualizer.
        
        Args:
            screen_width (int): Width of the video frames
            screen_height (int): Height of the video frames
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_center_x = screen_width / 2
    
    def _calculate_centroid(self, box):
        """
        Calculate the centroid (center point) of a bounding box.
        
        Args:
            box (np.array): Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            tuple: (center_x, center_y) centroid coordinates
        """
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        return (center_x, center_y)
    
    def _extract_player_features(self, feature_vector, player_offset=0):
        """
        Extract player features from the 360-element feature vector.
        
        Args:
            feature_vector (np.array): 360-element feature vector
            player_offset (int): Offset for player (0 for player 1, 180 for player 2)
            
        Returns:
            dict: Dictionary containing player features
        """
        # Feature vector structure per player:
        # 0: presence flag
        # 1-4: bounding box [x1, y1, x2, y2]
        # 5-6: centroid [cx, cy]
        # 7-8: velocity [vx, vy]
        # 9-10: acceleration [ax, ay]
        # 11: speed magnitude
        # 12: acceleration magnitude
        # 13-46: keypoints (17*2)
        # 47-63: keypoint confidences (17)
        # 64-97: keypoint velocities (17*2)
        # 98-131: keypoint accelerations (17*2)
        # 132-148: keypoint speeds (17)
        # 149-165: keypoint acceleration magnitudes (17)
        # 166-179: limb lengths (14)
        
        offset = player_offset * 180  # 180 features per player
        
        # Check if player is present
        if feature_vector[offset] != 1.0:
            return None
            
        player_features = {
            'presence': feature_vector[offset],
            'box': feature_vector[offset+1:offset+5],
            'centroid': feature_vector[offset+5:offset+7],
            'velocity': feature_vector[offset+7:offset+9],
            'acceleration': feature_vector[offset+9:offset+11],
            'speed': feature_vector[offset+11],
            'acceleration_magnitude': feature_vector[offset+12],
            'keypoints': feature_vector[offset+13:offset+47].reshape(17, 2),
            'keypoint_confidences': feature_vector[offset+47:offset+64],
            'keypoint_velocities': feature_vector[offset+64:offset+98].reshape(17, 2),
            'keypoint_accelerations': feature_vector[offset+98:offset+132].reshape(17, 2),
            'keypoint_speeds': feature_vector[offset+132:offset+149],
            'keypoint_acceleration_magnitudes': feature_vector[offset+149:offset+166],
            'limb_lengths': feature_vector[offset+166:offset+180]
        }
        
        return player_features
    
    def _draw_player_features(self, frame, player_features, color, label, position_offset=(0, 0)):
        """
        Draw player with feature values displayed as text.
        
        Args:
            frame (np.array): Video frame
            player_features (dict): Player features extracted from feature vector
            color (tuple): BGR color tuple for the player
            label (str): Label for the player
            position_offset (tuple): (x, y) offset for text positioning
            
        Returns:
            np.array: Frame with player and features drawn
        """
        if player_features is None:
            return frame
            
        # Draw bounding box
        box = player_features['box']
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw centroid
            centroid = player_features['centroid']
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 5, color, -1)
        
        # Draw feature values as text
        x_offset, y_offset = position_offset
        y_pos = y_offset
        
        # Draw player label
        cv2.putText(frame, f"{label} Features:", (x_offset, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['feature_text'], 2)
        y_pos += 25
        
        # Draw velocity
        velocity = player_features['velocity']
        if velocity is not None:
            vel_text = f"  Velocity: ({velocity[0]:.2f}, {velocity[1]:.2f})"
            cv2.putText(frame, vel_text, (x_offset, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['velocity'], 2)
            y_pos += 25
        
        # Draw acceleration
        acceleration = player_features['acceleration']
        if acceleration is not None:
            acc_text = f"  Acceleration: ({acceleration[0]:.2f}, {acceleration[1]:.2f})"
            cv2.putText(frame, acc_text, (x_offset, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['acceleration'], 2)
            y_pos += 25
        
        # Draw magnitudes
        speed = player_features['speed']
        accel_mag = player_features['acceleration_magnitude']
        if speed is not None and accel_mag is not None:
            mag_text = f"  Speed: {speed:.2f}, Accel Mag: {accel_mag:.2f}"
            cv2.putText(frame, mag_text, (x_offset, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['feature_text'], 2)
        
        return frame
    
    def _draw_point_indicator(self, frame, target_value, width):
        """
        Draw point indicator on the frame.
        
        Args:
            frame (np.array): Video frame
            target_value (int): Target value (-100, 0, or 1)
            width (int): Width of the frame
            
        Returns:
            np.array: Frame with point indicator drawn
        """
        if target_value == -100:
            # This is a skipped frame
            point_text = "SKIPPED FRAME"
            point_color = COLORS['skipped_indicator']
            position = (width // 2 - 100, 40)  # Top-center
            cv2.putText(frame, point_text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color, 1)
        else:
            # This is either in point (1) or not in point (0)
            is_in_point = target_value == 1
            point_text = "IN POINT" if is_in_point else "NOT IN POINT"
            point_color = COLORS['point_indicator'] if is_in_point else COLORS['not_point_indicator']
            position = (width // 2 - 80, 40)  # Top-center
            cv2.putText(frame, point_text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, point_color, 2)
        
        return frame
    
    def validate_and_visualize_features(self, preprocessed_npz_path, features_npz_path, 
                                      video_path, output_path, 
                                      start_time=0, duration=99999):
        """
        Validate engineered features by mapping feature vectors to original video frames.
        Combines preprocessed data with feature vectors for comprehensive validation.
        
        Args:
            preprocessed_npz_path (str): Path to preprocessed .npz file
            features_npz_path (str): Path to feature vectors .npz file
            video_path (str): Path to video file
            output_path (str): Path to output video file
            start_time (int): Start time in seconds
            duration (int): Duration in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Loading preprocessed data from: {preprocessed_npz_path}")
            preprocessed_data = np.load(preprocessed_npz_path, allow_pickle=True)
            
            print(f"Loading feature vectors from: {features_npz_path}")
            features_data = np.load(features_npz_path, allow_pickle=True)
            
            # Extract arrays from preprocessed data
            preprocessed_frames = preprocessed_data['frames']
            preprocessed_targets = preprocessed_data['targets']
            near_players = preprocessed_data['near_players']
            far_players = preprocessed_data['far_players']
            court_mask = preprocessed_data['court_mask'] if 'court_mask' in preprocessed_data else None
            
            # Extract arrays from feature data
            feature_vectors = features_data['features']
            feature_targets = features_data['targets']
            
            print(f"Preprocessed data: {len(preprocessed_frames)} frames")
            print(f"Feature vectors: {len(feature_vectors)} vectors")
            
            # Find annotated frames (not skipped)
            annotated_indices = np.where(preprocessed_targets != -100)[0]
            print(f"Annotated frames: {len(annotated_indices)}")
            
            # Verify mapping between preprocessed and feature data
            if len(annotated_indices) != len(feature_vectors):
                print(f"❌ Mismatch: {len(annotated_indices)} annotated frames vs {len(feature_vectors)} feature vectors")
                return False
            
            # Verify target alignment
            preprocessed_annotated_targets = preprocessed_targets[annotated_indices]
            if not np.array_equal(preprocessed_annotated_targets, feature_targets):
                print("❌ Target alignment mismatch between preprocessed and feature data")
                return False
            
            print("✅ Data mapping verified successfully")
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Error opening video file: {video_path}")
                return False
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate start and end frames
            start_frame = int(start_time * fps)
            end_frame = min(int((start_time + duration) * fps), total_frames)
            
            print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
            print(f"Processing frames: {start_frame} to {end_frame}")
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"❌ Error creating output video file: {output_path}")
                cap.release()
                return False
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Process frames
            frame_count = 0
            current_frame = start_frame
            feature_vector_index = 0
            
            # Find the first feature vector index that corresponds to our start frame
            while feature_vector_index < len(annotated_indices) and annotated_indices[feature_vector_index] < start_frame:
                feature_vector_index += 1
            
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Draw court mask if available
                if court_mask is not None:
                    overlay = frame.copy()
                    overlay[court_mask > 0] = [0, 255, 0]  # Green for court area
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                # Process annotated frames only
                if feature_vector_index < len(annotated_indices) and annotated_indices[feature_vector_index] == current_frame:
                    # Get feature vector for this frame
                    feature_vector = feature_vectors[feature_vector_index]
                    target_value = feature_targets[feature_vector_index]
                    
                    # DEBUG: Verify we're using actual feature data
                    if feature_vector_index < 5:  # Only for first few frames
                        p1_vel = feature_vector[7:9]
                        p1_acc = feature_vector[9:11]
                        print(f"    Feature Vector {feature_vector_index}: "
                              f"Velocity=({p1_vel[0]:.2f}, {p1_vel[1]:.2f}), "
                              f"Acceleration=({p1_acc[0]:.2f}, {p1_acc[1]:.2f})")
                    
                    # Extract player features from feature vector
                    player1_features = self._extract_player_features(feature_vector, player_offset=0)
                    player2_features = self._extract_player_features(feature_vector, player_offset=1)
                    
                    # Draw player 1 (near) features - top-right corner
                    if player1_features is not None:
                        frame = self._draw_player_features(
                            frame, player1_features, COLORS['player1'], "Player 1 (Near)",
                            position_offset=(width - 350, 30)
                        )
                    
                    # Draw player 2 (far) features - below Player 1 features
                    if player2_features is not None:
                        frame = self._draw_player_features(
                            frame, player2_features, COLORS['player2'], "Player 2 (Far)",
                            position_offset=(width - 350, 150)
                        )
                    
                    # Add point indicator
                    frame = self._draw_point_indicator(frame, target_value, width)
                    
                    # Move to next feature vector
                    feature_vector_index += 1
                elif feature_vector_index < len(annotated_indices) and annotated_indices[feature_vector_index] > current_frame:
                    # This is a skipped frame (not annotated)
                    frame = self._draw_point_indicator(frame, -100, width)
                
                # Write frame to output video
                out.write(frame)
                
                current_frame += 1
                frame_count += 1
                
                # Progress indicator
                if frame_count % 30 == 0:
                    print(f"  Processed {frame_count} frames")
            
            # Clean up
            cap.release()
            out.release()
            
            print(f"✅ Finished processing {frame_count} frames")
            print(f"Output saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error during feature visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Test the enhanced feature visualizer."""
    # Initialize visualizer
    visualizer = EnhancedFeatureVisualizer()
    
    # Example usage
    preprocessed_npz_path = "pose_data/preprocessed/yolos_0.25conf_15fps_0s_to_99999s/Satoru Nakajima (11.75 UTR) Match Video (Unedited) vs. Dylan Chou (11.14 UTR) November 5, 2023_preprocessed.npz"
    features_npz_path = "pose_data/features/yolos_0.25conf_15fps_0s_to_99999s/Satoru Nakajima (11.75 UTR) Match Video (Unedited) vs. Dylan Chou (11.14 UTR) November 5, 2023_features.npz"
    video_path = "raw_videos/Satoru Nakajima (11.75 UTR) Match Video (Unedited) vs. Dylan Chou (11.14 UTR) November 5, 2023.mp4"
    output_path = "sanity_check_clips/enhanced_feature_visualization_test.mp4"
    
    if os.path.exists(preprocessed_npz_path) and os.path.exists(features_npz_path) and os.path.exists(video_path):
        print("Testing enhanced feature visualization...")
        success = visualizer.validate_and_visualize_features(
            preprocessed_npz_path, features_npz_path, video_path, output_path,
            start_time=0, duration=60  # Short test duration
        )
        print(f"Visualization {'succeeded' if success else 'failed'}")
    else:
        print("Test files not found. Please ensure the files exist.")

if __name__ == "__main__":
    main()