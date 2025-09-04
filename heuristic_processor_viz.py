#!/usr/bin/env python3
"""
Visualization script for the heuristic processor.
Processes filtered pose data, applies the heuristic processor, and annotates videos with player assignments.
"""

import argparse
import os
import cv2
import numpy as np
from heuristic_processor import HeuristicProcessor

def draw_bounding_box_with_label(frame, box, label, color=(0, 255, 0)):
    """Draw a bounding box with a label on the frame."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def process_video_with_heuristic(input_dir, output_dir):
    """Process all filtered pose data files and create annotated videos."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the heuristic processor
    processor = HeuristicProcessor(screen_width=1280, screen_height=720)
    
    # Process each .npz file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.npz'):
            npz_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")
            
            # Extract video name from the npz filename
            # Remove the "_posedata_0s_to_999999s_yolos.npz" part
            video_name = filename.replace('_posedata_0s_to_999999s_yolos.npz', '.mp4')
            video_path = os.path.join('/Users/ismaelrobles-razzaq/Desktop/tennis_tracker/raw_videos', video_name)
            
            # Check if video exists
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                continue
                
            # Load the filtered pose data
            try:
                all_frames_data = np.load(npz_path, allow_pickle=True)['frames']
            except Exception as e:
                print(f"Error loading pose data from {npz_path}: {e}")
                continue
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video file: {video_path}")
                continue
                
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create output video writer
            output_video_path = os.path.join(output_dir, f"player_assigned_{video_name}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # Process each frame
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Resize frame to match expected dimensions (1280x720)
                if width != 1280 or height != 720:
                    frame = cv2.resize(frame, (1280, 720))
                
                # Get pose data for this frame if available
                if frame_idx < len(all_frames_data):
                    frame_data = all_frames_data[frame_idx]
                    
                    # Apply the heuristic processor
                    assigned_players = processor.assign_players(frame_data)
                    
                    # Draw bounding boxes with labels
                    if assigned_players['near_player'] is not None:
                        draw_bounding_box_with_label(
                            frame, 
                            assigned_players['near_player']['box'], 
                            "Near Player", 
                            (0, 0, 255)  # Red
                        )
                    
                    if assigned_players['far_player'] is not None:
                        draw_bounding_box_with_label(
                            frame, 
                            assigned_players['far_player']['box'], 
                            "Far Player", 
                            (255, 0, 0)  # Blue
                        )
                
                # Write the annotated frame to output video
                out.write(frame)
                frame_idx += 1
                
                # Print progress every 100 frames
                if frame_idx % 100 == 0:
                    print(f"Processed {frame_idx} frames...")
            
            # Release resources
            cap.release()
            out.release()
            print(f"Finished processing {filename}. Output saved to {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description="Process filtered pose data and annotate videos with player assignments")
    parser.add_argument("input_dir", help="Input directory containing filtered .npz pose data files")
    parser.add_argument("output_dir", help="Output directory for annotated videos")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Input directory does not exist: {args.input_dir}")
        return
    
    process_video_with_heuristic(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()