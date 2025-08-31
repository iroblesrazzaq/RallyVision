#!/usr/bin/env python3
"""
Pose Data Filtering Script

This script filters pose data from .npz files based on a playable area mask image.
It removes detections whose bounding box centroids fall outside the playable area.

Usage:
    python filter_pose_data.py --input-npz <input_file> --mask <mask_image> --output-npz <output_file> [--video-path <video>] [--visualize]

Arguments:
    --input-npz: Required. Path to the input pose data .npz file
    --mask: Required. Path to the playable area mask image (.png or .jpg)
    --output-npz: Required. Path where the filtered .npz file will be saved
    --video-path: Optional. Path to the original source video (required if --visualize is used)
    --visualize: Optional. Flag to create a verification video showing the filtering results
"""

import os
import sys
import argparse
import numpy as np
import cv2
import time


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Filter pose data based on playable area mask",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python filter_pose_data.py \\
        --input-npz "pose_data/yolos_0.05conf_15fps_0s_to_60s/my_video_posedata.npz" \\
        --mask "court_masks/my_video_mask.png" \\
        --output-npz "pose_data_filtered/yolos_0.05conf_15fps_0s_to_60s/my_video_posedata_filtered.npz"
    
    python filter_pose_data.py \\
        --input-npz "pose_data/yolos_0.05conf_15fps_0s_to_60s/my_video_posedata.npz" \\
        --mask "court_masks/my_video_mask.png" \\
        --output-npz "pose_data_filtered/yolos_0.05conf_15fps_0s_to_60s/my_video_posedata_filtered.npz" \\
        --video-path "raw_videos/my_video.mp4" \\
        --visualize
        """
    )
    
    parser.add_argument(
        '--input-npz',
        required=True,
        help='Path to the input pose data .npz file'
    )
    
    parser.add_argument(
        '--mask',
        required=True,
        help='Path to the playable area mask image (.png or .jpg)'
    )
    
    parser.add_argument(
        '--output-npz',
        required=True,
        help='Path where the filtered .npz file will be saved'
    )
    
    parser.add_argument(
        '--video-path',
        help='Path to the original source video (required if --visualize is used)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create a verification video showing the filtering results'
    )
    
    return parser.parse_args()


def create_visualization_video(video_path, output_video_path, filtered_data, mask):
    """
    Create a verification video showing the filtering results.
    
    Args:
        video_path (str): Path to the original source video
        output_video_path (str): Path where the output video will be saved
        filtered_data (list): List of filtered frame data
        mask (np.ndarray): The playable area mask
    """
    # Open the source video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video: {output_video_path}")
        cap.release()
        return
    
    # Create a color version of the mask for overlay
    mask_color = np.zeros((height, width, 3), dtype=np.uint8)
    mask_color[mask != 0] = [0, 255, 0]  # Green for playable area
    
    frame_count = 0
    print("Creating visualization video...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Get the corresponding filtered frame data
        if frame_count < len(filtered_data):
            frame_pose_data = filtered_data[frame_count]
        else:
            # If we run out of pose data, use empty data
            frame_pose_data = {
                'boxes': np.array([]),
                'keypoints': np.array([]),
                'conf': np.array([])
            }
        
        # Overlay the mask on the frame (semi-transparent)
        mask_overlay = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)
        
        # Draw annotations if this frame has pose data
        if len(frame_pose_data['boxes']) > 0 or len(frame_pose_data['keypoints']) > 0:
            _draw_annotations(mask_overlay, frame_pose_data)
        
        # Write frame to video
        out.write(mask_overlay)
        
        # Update progress
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    # Cleanup
    cap.release()
    out.release()
    print(f"✓ Visualization video saved to: {output_video_path}")


def _draw_annotations(frame, frame_pose_data):
    """
    Draw bounding boxes and keypoints on a frame.
    
    Args:
        frame: OpenCV frame to draw on
        frame_pose_data: Dictionary containing 'boxes', 'keypoints', and 'conf'
    """
    boxes = frame_pose_data['boxes']
    keypoints = frame_pose_data['keypoints']
    confidences = frame_pose_data['conf']
    
    # Draw bounding boxes and keypoints for each detected person
    for person_idx in range(len(boxes)):
        # Draw bounding box
        if len(boxes) > 0:
            box = boxes[person_idx]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw keypoints
        if len(keypoints) > 0 and len(confidences) > 0:
            person_keypoints = keypoints[person_idx]
            person_conf = confidences[person_idx]
            
            for kp_idx, (kp_x, kp_y) in enumerate(person_keypoints):
                if person_conf[kp_idx] >= 0.5:  # Keypoint confidence threshold
                    cv2.circle(frame, (int(kp_x), int(kp_y)), 3, (0, 0, 255), -1)


def main():
    """
    Main function to filter pose data based on playable area mask.
    """
    # Parse command-line arguments
    args = parse_args()
    
    print("=== Pose Data Filtering ===")
    print(f"Input pose data: {args.input_npz}")
    print(f"Mask image: {args.mask}")
    print(f"Output pose data: {args.output_npz}")
    if args.visualize:
        print(f"Video path: {args.video_path}")
        print("Visualization: Enabled")
    print()
    
    # Load the mask image
    try:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error: Could not load mask image: {args.mask}")
            sys.exit(1)
        print(f"✓ Loaded mask image: {mask.shape}")
    except FileNotFoundError:
        print(f"Error: Mask file not found: {args.mask}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading mask image: {e}")
        sys.exit(1)
    
    # Load the pose data
    try:
        pose_data = np.load(args.input_npz, allow_pickle=True)['frames']
        print(f"✓ Loaded pose data: {len(pose_data)} frames")
    except FileNotFoundError:
        print(f"Error: Pose data file not found: {args.input_npz}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading pose data: {e}")
        sys.exit(1)
    
    # Initialize data structures for filtered results
    filtered_frames_data = []
    
    print("Filtering pose data...")
    
    # Main filtering loop
    for frame_idx, frame_data in enumerate(pose_data):
        # Extract arrays from frame data
        boxes = frame_data['boxes']
        keypoints = frame_data['keypoints']
        conf = frame_data['conf']
        
        # Initialize empty lists for the current frame's surviving data
        kept_boxes = []
        kept_keypoints = []
        kept_conf = []
        
        # Inner loop: filter each person in the frame
        for i, box in enumerate(boxes):
            # Calculate bounding box centroid
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            
            # Check if centroid is inside the playable area
            # First, perform boundary check to prevent IndexError
            if (0 <= center_y < mask.shape[0] and 
                0 <= center_x < mask.shape[1] and 
                mask[int(center_y), int(center_x)] != 0):  # Non-zero means "inside"
                
                # Keep this person's data
                kept_boxes.append(box)
                kept_keypoints.append(keypoints[i])
                kept_conf.append(conf[i])
        
        # Assemble new frame data
        filtered_frame = {
            'boxes': np.array(kept_boxes),
            'keypoints': np.array(kept_keypoints),
            'conf': np.array(kept_conf)
        }
        filtered_frames_data.append(filtered_frame)
        
        # Update progress
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{len(pose_data)} frames...")
    
    print(f"✓ Filtering complete: {len(filtered_frames_data)} frames processed")
    
    # Save the filtered data
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output_npz)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the filtered data
        np.savez_compressed(args.output_npz, frames=filtered_frames_data)
        print(f"✓ Filtered pose data saved to: {args.output_npz}")
    except Exception as e:
        print(f"Error saving filtered data: {e}")
        sys.exit(1)
    
    # Create visualization if requested
    if args.visualize:
        if not args.video_path:
            print("Error: --video-path is required for visualization.")
            sys.exit(1)
        
        # Define output video path
        base_name = os.path.splitext(os.path.basename(args.output_npz))[0]
        output_video_path = f"sanity_check_clips/{base_name}_filtered_visualization.mp4"
        
        create_visualization_video(args.video_path, output_video_path, filtered_frames_data, mask)
        print(f"✓ Visualization video saved to: {output_video_path}")
    
    print("=== Filtering Complete ===")


if __name__ == "__main__":
    # Start timing
    script_start_time = time.time()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Calculate and print total runtime
    script_end_time = time.time()
    total_runtime = script_end_time - script_start_time
    
    print(f"\n🎯 Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
