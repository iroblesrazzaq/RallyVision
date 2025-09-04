import cv2
import torch
import sys
import time
import os
import numpy as np
from ultralytics import YOLO


class OptimizedPoseExtractor:
    """
    An optimized version of PoseExtractor that uses batch processing 
    to leverage Apple MPS parallel inference capabilities.
    """
    
    def __init__(self, model_path='yolov8n-pose.pt', batch_size=8):
        """
        Initialize the OptimizedPoseExtractor with a YOLOv8-pose model.
        
        Args:
            model_path (str): Path to the YOLOv8-pose model file.
                             Defaults to 'yolov8s-pose.pt'
            batch_size (int): Number of frames to process in parallel.
                             Defaults to 8.
        """
        # Try to use MPS if available, fallback to CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
            print(f"Using device: {self.device.upper()} (MPS available)")
        else:
            self.device = "cpu"
            print(f"Using device: {self.device.upper()} (MPS not available, using CPU)")
        
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.batch_size = batch_size
        print(f"YOLOv8-pose model loaded successfully from: {model_path}")
        print(f"Batch size for parallel processing: {batch_size}")
    
    def _process_batch(self, frames_batch, confidence_threshold):
        """
        Process a batch of frames using the model.
        
        Args:
            frames_batch (list): List of frames to process
            confidence_threshold (float): Confidence threshold for the model
            
        Returns:
            list: List of results for each frame in the batch
        """
        if not frames_batch:
            return []
            
        # Process all frames in the batch together
        results = self.model(
            frames_batch, 
            verbose=False, 
            device=self.device, 
            conf=confidence_threshold,
            imgsz=1920
        )
        
        return results
    
    def _extract_frame_data(self, result):
        """
        Extract raw numerical data from a single frame result.
        
        Args:
            result: YOLO result for a single frame
            
        Returns:
            dict: Dictionary containing boxes, keypoints, and confidences
        """
        frame_data = {}
        
        if result.boxes is not None:
            # Extract bounding boxes
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
            frame_data['boxes'] = boxes
            
            # Extract keypoints
            keypoints = result.keypoints.xy.cpu().numpy()  # [num_persons, num_keypoints, 2]
            frame_data['keypoints'] = keypoints
            
            # Extract keypoint confidences
            keypoint_conf = result.keypoints.conf.cpu().numpy()  # [num_persons, num_keypoints]
            frame_data['conf'] = keypoint_conf
        else:
            # No detections in this frame
            frame_data['boxes'] = np.array([])
            frame_data['keypoints'] = np.array([])
            frame_data['conf'] = np.array([])
            
        return frame_data
    
    def extract_pose_data(self, video_path, confidence_threshold, start_time_seconds=0, duration_seconds=60, target_fps=15):
        """
        Extract raw pose data from a video segment and save to .npz file.
        Uses batch processing to leverage parallel inference on Apple MPS.
        
        Args:
            video_path (str): Path to the input video file
            confidence_threshold (float): Confidence threshold for the model - no default
            start_time_seconds (int): Start time in seconds (default: 0)
            duration_seconds (int): Duration to process in seconds (default: 60)
            target_fps (int): Target frame rate for consistent temporal sampling (default: 15)
            
        Returns:
            str: Path to the created .npz file
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range
        start_frame = int(start_time_seconds * fps)
        end_frame = min(int(start_frame + (duration_seconds * fps)), total_frames)
        
        print(f"Processing frames {start_frame} to {end_frame} (Source FPS: {fps})")
        print(f"Target FPS: {target_fps}")
        
        # Calculate frame selection for target FPS
        frame_interval = fps / target_fps
        print(f"Frame interval: {frame_interval:.2f} frames")
        
        # Set video position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Initialize list to store all frame data
        all_frames_data = []
        total_frames_to_process = end_frame - start_frame
        
        print("Extracting pose data...")
        processed_frames = 0
        
        # Calculate which frames to process for target FPS
        target_frames = []
        frame_interval = fps / target_fps
        for i in range(total_frames_to_process):
            current_time = start_time_seconds + (i / fps)
            target_frame_index = int(current_time * target_fps)
            target_time = target_frame_index / target_fps
            
            # Check if this frame should be processed (closest to target time)
            if abs(current_time - target_time) <= (1 / target_fps) / 2:
                target_frames.append(i)
        
        print(f"Will process {len(target_frames)} frames out of {total_frames_to_process} total frames")
        
        # Prepare for batch processing
        # Pre-fill all_frames_data with placeholders
        for _ in range(len(target_frames)):
            all_frames_data.append({
                'boxes': np.array([]),
                'keypoints': np.array([]),
                'conf': np.array([])
            })
        
        # Keep track of the index in target_frames for processed frames
        target_frame_idx = 0
        frames_to_process = []
        
        for i in range(total_frames_to_process):
            ret, frame = cap.read()
            
            if not ret:
                print(f"Error reading frame {i + start_frame}")
                break
            
            # Check if we should process this frame
            if i in target_frames:
                # Add frame to batch
                frames_to_process.append(frame)
                
                # Process in batches when we have enough frames or at the end
                if len(frames_to_process) >= self.batch_size or target_frame_idx == len(target_frames) - 1:
                    # Process the batch
                    batch_results = self._process_batch(frames_to_process, confidence_threshold)
                    
                    # Extract data from results and place in correct positions
                    for j, result in enumerate(batch_results):
                        frame_data = self._extract_frame_data(result)
                        # Place at the correct position in all_frames_data
                        all_frames_data[target_frame_idx - len(frames_to_process) + 1 + j] = frame_data
                    
                    processed_frames += len(frames_to_process)
                    
                    # Update progress bar
                    progress = (i + 1) / total_frames_to_process * 100
                    print(f"\rProgress: [{('=' * int(progress/2)).ljust(50)}] {progress:.1f}% (processed {processed_frames} frames)", end='', flush=True)
                    
                    # Reset for next batch
                    frames_to_process = []
                
                target_frame_idx += 1
        
        print()  # New line after progress bar
        
        # Cleanup
        cap.release()
        
        print(f"✓ Successfully extracted pose data from {len(all_frames_data)} frames")
        
        # Save data to .npz file
        # Extract model size from model path (e.g., "yolov8s-pose.pt" -> "s")
        if 'yolov8' in self.model_path:
            model_size = self.model_path.split('yolov8')[1].split('-')[0]
        else:
            model_size = 's'  # Default fallback
        
        # Create subdirectory with model size, confidence threshold, fps, and time range
        subdir_name = f"yolo{model_size}_{confidence_threshold}conf_{target_fps}fps_{start_time_seconds}s_to_{start_time_seconds + duration_seconds}s"
        output_dir = os.path.join("pose_data", "unfiltered", subdir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct descriptive filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}_posedata_{start_time_seconds}s_to_{start_time_seconds + duration_seconds}s_yolo{model_size}_optimized.npz"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save compressed data
        np.savez_compressed(output_path, frames=all_frames_data)
        print(f"✓ Pose data saved to: {output_path}")
        
        return output_path


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) >= 3:
        start_time = int(sys.argv[1])
        duration = int(sys.argv[2])
        target_fps = int(sys.argv[3]) if len(sys.argv) > 3 else 15
        confidence_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
        video_path = sys.argv[5] if len(sys.argv) > 5 else "raw_videos/Monica Greene unedited tennis match play.mp4"
        model_size = sys.argv[6] if len(sys.argv) > 6 else "s"
        batch_size = int(sys.argv[7]) if len(sys.argv) > 7 else 8
    else:
        start_time = 0
        duration = 10  # Default to 10 seconds for testing
        target_fps = 15
        confidence_threshold = 0.05
        video_path = "raw_videos/Monica Greene unedited tennis match play.mp4"
        model_size = "s"
        batch_size = 8
    
    # Start timing
    script_start_time = time.time()
    
    # Construct model path based on model size
    model_path = f"yolov8{model_size}-pose.pt"
    
    print("Initializing OptimizedPoseExtractor...")
    pose_extractor = OptimizedPoseExtractor(model_path=model_path, batch_size=batch_size)
    
    print(f"Processing video: {video_path}")
    print(f"Start time: {start_time}s, Duration: {duration}s, Target FPS: {target_fps}, Model: {model_path}")
    print(f"Batch size: {batch_size}")
    
    # Extract pose data
    output_path = pose_extractor.extract_pose_data(
        video_path=video_path,
        start_time_seconds=start_time,
        duration_seconds=duration,
        target_fps=target_fps,
        confidence_threshold=confidence_threshold
    )
    
    if output_path is None:
        print("Failed to extract pose data. Exiting.")
        exit()
    
    # Calculate and print total runtime
    script_end_time = time.time()
    total_runtime = script_end_time - script_start_time
    
    print(f"\n🎯 Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")