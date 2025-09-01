#!/usr/bin/env python3
"""
YOLO Player Detector for Tennis Videos
Uses YOLO object detection to recognize players in tennis videos.
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch

class YOLOPlayerDetector:
    def __init__(self, model_size='n'):
        """
        Initialize YOLO player detector
        
        Args:
            model_size (str): Model size - 'n', 's', 'm', or 'l'
        """
        self.model_size = model_size
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load YOLO model
        self.load_model()
        
    def load_model(self):
        """Load YOLO model based on size parameter"""
        model_path = f"yolov8{self.model_size}.pt"
        
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found. Downloading...")
            self.model = YOLO(f"yolov8{self.model_size}")
        else:
            print(f"Loading existing model: {model_path}")
            self.model = YOLO(model_path)
            
        print(f"Model loaded successfully: {self.model_size}")
    
    def detect_players(self, video_path, output_dir, duration_minutes=2):
        """
        Detect players in video and save annotated version
        
        Args:
            video_path (str): Path to input video
            output_dir (str): Directory to save annotated video
            duration_minutes (int): Duration to process in minutes
        """
        print(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frames to process (2 minutes)
        frames_to_process = min(duration_minutes * 60 * fps, total_frames)
        start_frame = 0  # Start from beginning
        
        print(f"Video: {fps} FPS, {width}x{height}, {total_frames} frames")
        print(f"Processing {frames_to_process} frames ({duration_minutes} minutes)")
        
        # Create output video writer
        output_path = os.path.join(output_dir, f"{Path(video_path).stem}_yolo_{self.model_size}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_frames = 0
        
        while cap.isOpened() and processed_frames < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip frames until we reach start_frame
            if frame_count < start_frame:
                continue
                
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            # Draw bounding boxes for person detections
            annotated_frame = self.draw_detections(frame, results)
            
            # Write frame to output video
            out.write(annotated_frame)
            processed_frames += 1
            
            # Progress indicator
            if processed_frames % (fps * 10) == 0:  # Every 10 seconds
                print(f"Processed {processed_frames}/{frames_to_process} frames")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"Annotated video saved to: {output_path}")
        return output_path
    
    def draw_detections(self, frame, results):
        """Draw bounding boxes and labels on frame"""
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence and class
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[cls]
                    
                    # Only draw person detections (class 0 in COCO dataset)
                    if class_name == 'person' and conf > 0.3:  # Confidence threshold
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"Player: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame
    
    def process_all_videos(self, input_dir, output_dir, duration_minutes=2):
        """Process all videos in input directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(input_path.glob(f"*{ext}"))
        
        print(f"Found {len(video_files)} video files")
        
        processed_videos = []
        for video_file in video_files:
            try:
                output_video = self.detect_players(
                    str(video_file), 
                    str(output_path), 
                    duration_minutes
                )
                processed_videos.append(output_video)
                print(f"Successfully processed: {video_file.name}")
            except Exception as e:
                print(f"Error processing {video_file.name}: {e}")
        
        print(f"\nProcessing complete! {len(processed_videos)} videos processed.")
        return processed_videos

def main():
    parser = argparse.ArgumentParser(description='YOLO Player Detector for Tennis Videos')
    parser.add_argument('--model-size', '-m', choices=['n', 's', 'm', 'l'], 
                       default='n', help='YOLO model size (n=nano, s=small, m=medium, l=large)')
    parser.add_argument('--input-dir', '-i', default='../raw_videos',
                       help='Input directory containing videos (default: ../raw_videos)')
    parser.add_argument('--output-dir', '-o', default='./annotated_videos',
                       help='Output directory for annotated videos (default: ./annotated_videos)')
    parser.add_argument('--duration', '-d', type=int, default=2,
                       help='Duration to process in minutes (default: 2)')
    parser.add_argument('--single-video', '-v', 
                       help='Process single video file instead of directory')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YOLOPlayerDetector(args.model_size)
    
    if args.single_video:
        # Process single video
        if not os.path.exists(args.single_video):
            print(f"Error: Video file {args.single_video} not found")
            return
        
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        detector.detect_players(args.single_video, str(output_path), args.duration)
    else:
        # Process all videos in directory
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory {args.input_dir} not found")
            return
        
        detector.process_all_videos(args.input_dir, args.output_dir, args.duration)

if __name__ == "__main__":
    main()
