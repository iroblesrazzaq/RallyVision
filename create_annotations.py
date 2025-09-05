#!/usr/bin/env python3
"""
Script to help create annotation files for tennis point detection.

This script provides utilities to create and manage annotation files
that define when points occur in tennis videos.
"""

import pandas as pd
import os
import argparse

def create_empty_annotation_file(video_name, output_dir="./annotations"):
    """
    Create an empty annotation file for a video.
    
    Args:
        video_name (str): Name of the video (without extension)
        output_dir (str): Directory to save the annotation file
        
    Returns:
        str: Path to the created annotation file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create empty DataFrame with required columns
    df = pd.DataFrame(columns=['start_frame', 'end_frame', 'label'])
    
    # Save to CSV
    filename = f"{video_name}_annotations.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    
    print(f"Created empty annotation file: {filepath}")
    print("Columns:")
    print("  - start_frame: Frame number where the point starts")
    print("  - end_frame: Frame number where the point ends")
    print("  - label: Label for the segment (e.g., 'point')")
    print("\nExample rows to add:")
    print("  100,250,point")
    print("  400,550,point")
    print("  720,850,point")
    
    return filepath

def add_annotation(filepath, start_frame, end_frame, label="point"):
    """
    Add an annotation to an existing annotation file.
    
    Args:
        filepath (str): Path to the annotation CSV file
        start_frame (int): Start frame of the point
        end_frame (int): End frame of the point
        label (str): Label for the point (default: "point")
    """
    # Load existing annotations
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        df = pd.DataFrame(columns=['start_frame', 'end_frame', 'label'])
    
    # Add new annotation
    new_row = pd.DataFrame({
        'start_frame': [start_frame],
        'end_frame': [end_frame],
        'label': [label]
    })
    
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Save updated annotations
    df.to_csv(filepath, index=False)
    print(f"Added annotation to {filepath}: {start_frame}-{end_frame} ({label})")

def list_annotations(filepath):
    """
    List all annotations in a file.
    
    Args:
        filepath (str): Path to the annotation CSV file
    """
    if not os.path.exists(filepath):
        print(f"Annotation file {filepath} does not exist")
        return
    
    df = pd.read_csv(filepath)
    
    if df.empty:
        print(f"No annotations in {filepath}")
        return
    
    print(f"Annotations in {filepath}:")
    for idx, row in df.iterrows():
        print(f"  {idx+1}. Frames {row['start_frame']}-{row['end_frame']} ({row['label']})")

def main():
    parser = argparse.ArgumentParser(description='Create and manage tennis point annotations')
    parser.add_argument('--create', type=str, 
                        help='Create empty annotation file for a video')
    parser.add_argument('--add', type=str, 
                        help='Add annotation to file (format: filepath,start_frame,end_frame,label)')
    parser.add_argument('--list', type=str, 
                        help='List all annotations in a file')
    parser.add_argument('--output-dir', type=str, default="./annotations",
                        help='Output directory for annotation files')
    
    args = parser.parse_args()
    
    if args.create:
        create_empty_annotation_file(args.create, args.output_dir)
    elif args.add:
        # Parse the add argument
        parts = args.add.split(',')
        if len(parts) >= 4:
            filepath, start_frame, end_frame, label = parts[0], int(parts[1]), int(parts[2]), parts[3]
            add_annotation(filepath, start_frame, end_frame, label)
        elif len(parts) == 3:
            filepath, start_frame, end_frame = parts[0], int(parts[1]), int(parts[2])
            add_annotation(filepath, start_frame, end_frame)
        else:
            print("Invalid format for --add. Use: filepath,start_frame,end_frame,label")
    elif args.list:
        list_annotations(args.list)
    else:
        print("No action specified. Use --help for usage information.")

if __name__ == "__main__":
    main()