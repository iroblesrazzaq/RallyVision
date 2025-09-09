#!/usr/bin/env python3
"""
Feature Visualizer for Tennis Pose Data

This module contains the FeatureVisualizer class that handles
visualization of engineered features on videos, including
displaying velocity and acceleration values as text.
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

class FeatureVisualizer:
    """
    Visualizes engineered features on videos, including displaying 
    velocity and acceleration values as text.
    """
    
    def __init__(self, screen_width=1280, screen_height=720):
        """
        Initialize the feature visualizer.
        
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
            tuple: (center_x, center_y) coordinates of the centroid
        """
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        return (center_x, center_y)
    
    def _draw_player_with_features(self, frame, player_data, color, label, velocity=None, acceleration=None):
        """
        Draw a player with feature values displayed as text.
        
        Args:
            frame (np.array): Video frame
            player_data (dict): Player data with 'box', 'keypoints', 'conf'
            color (tuple): BGR color tuple for the player
            label (str): Label for the player
            velocity (tuple): Optional velocity vector (vx, vy)
            acceleration (tuple): Optional acceleration vector (ax, ay)
            
        Returns:
            np.array: Frame with player and features drawn
        """
        if player_data is None:
            return frame
        
        # Draw bounding box
        box = player_data.get('box', None)
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw centroid
            centroid = self._calculate_centroid(box)
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 5, color, -1)
        
        # Draw keypoints
        kps = player_data.get('keypoints', None)
        if kps is not None:
            for x, y in kps:
                if x > 0 and y > 0:  # Only draw valid keypoints
                    cv2.circle(frame, (int(x), int(y)), 3, color, -1)
            
            # Draw connections
            for start_idx, end_idx in COCO_CONNECTIONS:
                if start_idx < len(kps) and end_idx < len(kps):
                    start_point = kps[start_idx]
                    end_point = kps[end_idx]
                    if start_point[0] > 0 and start_point[1] > 0 and end_point[0] > 0 and end_point[1] > 0:
                        cv2.line(frame, 
                                (int(start_point[0]), int(start_point[1])),
                                (int(end_point[0]), int(end_point[1])),
                                color, 2)
        
        return frame
    
    def _draw_feature_text(self, frame, player_label, velocity, acceleration, speed=None, acceleration_magnitude=None, position=None):
        """
        Draw velocity, acceleration, and magnitude values as text on the frame.
        
        Args:
            frame (np.array): Video frame
            player_label (str): Label for the player
            velocity (tuple): Velocity vector (vx, vy)
            acceleration (tuple): Acceleration vector (ax, ay)
            speed (float): Speed magnitude (optional)
            acceleration_magnitude (float): Acceleration magnitude (optional)
            position (tuple): (x, y) position for text display
            
        Returns:
            np.array: Frame with feature text drawn
        """
        x, y = position
        
        # Draw player label
        cv2.putText(frame, f"{player_label} Features:", (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['feature_text'], 2)
        
        # Draw velocity
        if velocity is not None:
            vel_text = f"  Velocity: ({velocity[0]:.2f}, {velocity[1]:.2f})"
            cv2.putText(frame, vel_text, (x, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['velocity'], 2)
        
        # Draw acceleration
        if acceleration is not None:
            acc_text = f"  Acceleration: ({acceleration[0]:.2f}, {acceleration[1]:.2f})"
            cv2.putText(frame, acc_text, (x, y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['acceleration'], 2)
        
        # Draw magnitudes if provided
        if speed is not None and acceleration_magnitude is not None:
            mag_text = f"  Speed: {speed:.2f}, Accel Mag: {acceleration_magnitude:.2f}"
            cv2.putText(frame, mag_text, (x, y + 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['feature_text'], 2)
        
        return frame
    
    def visualize_features_on_video(self, preprocessed_npz_path, video_path, output_path, 
                                   start_time=0, duration=99999, show_velocity=True, show_acceleration=True):
        """
        Visualize engineered features on a video, displaying velocity and acceleration as text.
        
        Args:
            preprocessed_npz_path (str): Path to preprocessed .npz file
            video_path (str): Path to video file
            output_path (str): Path to output video file
            start_time (int): Start time in seconds
            duration (int): Duration in seconds
            show_velocity (bool): Whether to show velocity values
            show_acceleration (bool): Whether to show acceleration values
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Loading preprocessed data from: {preprocessed_npz_path}")
            data = np.load(preprocessed_npz_path, allow_pickle=True)
            
            # Extract arrays
            frames = data['frames']
            targets = data['targets']
            near_players = data['near_players']
            far_players = data['far_players']
            court_mask = data['court_mask'] if 'court_mask' in data else None
            
            print(f"Loaded {len(frames)} frames")
            
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
            previous_near_player = None
            previous_far_player = None
            
            # Store feature values for display
            near_player_velocity = (0.0, 0.0)
            near_player_acceleration = (0.0, 0.0)
            far_player_velocity = (0.0, 0.0)
            far_player_acceleration = (0.0, 0.0)
            
            # Initialize previous velocity tracking for acceleration calculation
            if hasattr(self, '_previous_near_velocity'):
                delattr(self, '_previous_near_velocity')
            if hasattr(self, '_previous_far_velocity'):
                delattr(self, '_previous_far_velocity')
            
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Draw court mask if available
                if court_mask is not None:
                    overlay = frame.copy()
                    overlay[court_mask > 0] = [0, 255, 0]  # Green for court area
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                # Draw players with features for this frame
                if current_frame < len(near_players) and current_frame < len(far_players):
                    # Get current players
                    near_player = near_players[current_frame]
                    far_player = far_players[current_frame]
                    
                    # Calculate and store velocity/acceleration for near player
                    if near_player is not None and show_velocity:
                        if previous_near_player is not None:
                            current_centroid = self._calculate_centroid(near_player['box'])
                            previous_centroid = self._calculate_centroid(previous_near_player['box'])
                            current_velocity = (current_centroid[0] - previous_centroid[0],
                                               current_centroid[1] - previous_centroid[1])
                            
                            # Calculate acceleration as change in velocity
                            if hasattr(self, '_previous_near_velocity'):
                                near_player_acceleration = (current_velocity[0] - self._previous_near_velocity[0],
                                                           current_velocity[1] - self._previous_near_velocity[1])
                            else:
                                near_player_acceleration = (0.0, 0.0)
                            
                            # Store current velocity for next iteration
                            self._previous_near_velocity = current_velocity
                            near_player_velocity = current_velocity
                        else:
                            # Reset velocity/acceleration tracking when no previous player
                            if hasattr(self, '_previous_near_velocity'):
                                delattr(self, '_previous_near_velocity')
                            near_player_velocity = (0.0, 0.0)
                            near_player_acceleration = (0.0, 0.0)
                        
                        frame = self._draw_player_with_features(
                            frame, near_player, COLORS['player1'], "Player 1 (Near)"
                        )
                    elif near_player is not None:
                        frame = self._draw_player_with_features(
                            frame, near_player, COLORS['player1'], "Player 1 (Near)"
                        )
                    
                    # Calculate and store velocity/acceleration for far player
                    if far_player is not None and show_velocity:
                        if previous_far_player is not None:
                            current_centroid = self._calculate_centroid(far_player['box'])
                            previous_centroid = self._calculate_centroid(previous_far_player['box'])
                            current_velocity = (current_centroid[0] - previous_centroid[0],
                                               current_centroid[1] - previous_centroid[1])
                            
                            # Calculate acceleration as change in velocity
                            if hasattr(self, '_previous_far_velocity'):
                                far_player_acceleration = (current_velocity[0] - self._previous_far_velocity[0],
                                                          current_velocity[1] - self._previous_far_velocity[1])
                            else:
                                far_player_acceleration = (0.0, 0.0)
                            
                            # Store current velocity for next iteration
                            self._previous_far_velocity = current_velocity
                            far_player_velocity = current_velocity
                        else:
                            # Reset velocity/acceleration tracking when no previous player
                            if hasattr(self, '_previous_far_velocity'):
                                delattr(self, '_previous_far_velocity')
                            far_player_velocity = (0.0, 0.0)
                            far_player_acceleration = (0.0, 0.0)
                        
                        frame = self._draw_player_with_features(
                            frame, far_player, COLORS['player2'], "Player 2 (Far)"
                        )
                    elif far_player is not None:
                        frame = self._draw_player_with_features(
                            frame, far_player, COLORS['player2'], "Player 2 (Far)"
                        )
                    
                    # Display feature values as text
                    # Player 1 (Near) features - top-right corner
                    if near_player is not None:
                        # For visualization, we'll calculate magnitudes on the fly
                        speed = np.sqrt(near_player_velocity[0]**2 + near_player_velocity[1]**2) if show_velocity else None
                        accel_mag = np.sqrt(near_player_acceleration[0]**2 + near_player_acceleration[1]**2) if show_acceleration else None
                        
                        frame = self._draw_feature_text(
                            frame, "Player 1 (Near)", 
                            near_player_velocity if show_velocity else None,
                            near_player_acceleration if show_acceleration else None,
                            speed if show_velocity else None,
                            accel_mag if show_acceleration else None,
                            (width - 350, 30)
                        )
                    
                    # Player 2 (Far) features - below Player 1 features
                    if far_player is not None:
                        # For visualization, we'll calculate magnitudes on the fly
                        speed = np.sqrt(far_player_velocity[0]**2 + far_player_velocity[1]**2) if show_velocity else None
                        accel_mag = np.sqrt(far_player_acceleration[0]**2 + far_player_acceleration[1]**2) if show_acceleration else None
                        
                        frame = self._draw_feature_text(
                            frame, "Player 2 (Far)", 
                            far_player_velocity if show_velocity else None,
                            far_player_acceleration if show_acceleration else None,
                            speed if show_velocity else None,
                            accel_mag if show_acceleration else None,
                            (width - 350, 120)
                        )
                    
                    # Update previous players
                    previous_near_player = near_player
                    previous_far_player = far_player
                    
                    # Reset velocity tracking when players disappear
                    if near_player is None and hasattr(self, '_previous_near_velocity'):
                        delattr(self, '_previous_near_velocity')
                    if far_player is None and hasattr(self, '_previous_far_velocity'):
                        delattr(self, '_previous_far_velocity')
                    
                    # Add point indicator
                    if current_frame < len(targets):
                        target_value = targets[current_frame]
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
                            # Position: top-left for main indicator
                            position = (20, 40)
                            cv2.putText(frame, point_text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, point_color, 2)
                
                # Write frame to output video
                out.write(frame)
                
                frame_count += 1
                current_frame += 1
                
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
            print(f"❌ Error processing {preprocessed_npz_path}: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function for testing the visualizer."""
    # Example usage
    visualizer = FeatureVisualizer()
    
    # Test with a sample file
    npz_path = "pose_data/preprocessed/yolos_0.25conf_15fps_0s_to_99999s/Satoru Nakajima (11.75 UTR) Match Video (Unedited) vs. Dylan Chou (11.14 UTR) November 5, 2023_preprocessed.npz"
    video_path = "raw_videos/Satoru Nakajima (11.75 UTR) Match Video (Unedited) vs. Dylan Chou (11.14 UTR) November 5, 2023.mp4"
    output_path = "sanity_check_clips/feature_visualization_test.mp4"
    
    if os.path.exists(npz_path) and os.path.exists(video_path):
        print("Testing feature visualization...")
        success = visualizer.visualize_features_on_video(
            npz_path, video_path, output_path,
            start_time=0, duration=10,  # Short test duration
            draw_velocity=True, draw_acceleration=True
        )
        print(f"Visualization {'succeeded' if success else 'failed'}")
    else:
        print("Test files not found. Please ensure the files exist.")

if __name__ == "__main__":
    main()