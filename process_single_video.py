#!/usr/bin/env python3
"""
CLI tool to process a single video file with the heuristic processor.
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

def process_single_video(npz_path, video_path, output_path):
    """Process a single video file with the heuristic processor."""
    print(f"Processing {video_path}...")
    
    # Initialize the heuristic processor
    processor = HeuristicProcessor(screen_width=1280, screen_height=720)
    
    # Load the filtered pose data
    try:
        all_frames_data = np.load(npz_path, allow_pickle=True)['frames']
    except Exception as e:
        print(f"Error loading pose data from {npz_path}: {e}")
        return
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
        
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))  # Always output 1280x720
    
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
    print(f"Finished processing. Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Process a single video file with the heuristic processor")
    parser.add_argument("npz_path", help="Path to the filtered pose data .npz file")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("output_path", help="Path for the output annotated video file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.npz_path):
        print(f"Pose data file does not exist: {args.npz_path}")
        return
        
    if not os.path.exists(args.video_path):
        print(f"Video file does not exist: {args.video_path}")
        return
    
    process_single_video(args.npz_path, args.video_path, args.output_path)

if __name__ == "__main__":
    main()