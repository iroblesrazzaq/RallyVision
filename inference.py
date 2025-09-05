#!/usr/bin/env python3
"""
inference.py - Inference script for the TennisPointDetector model
"""

import torch
import numpy as np
from model import load_model

def load_pose_data(npz_path):
    """
    Load pose data from .npz file.
    
    Args:
        npz_path (str): Path to the .npz file
        
    Returns:
        np.array: Pose data array
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        frames = data['frames']
        
        # Convert to the format expected by the model
        # This is a simplified example - you'll need to adapt based on your actual data structure
        feature_vectors = []
        for frame_data in frames:
            # This is a placeholder - you'll need to implement the actual feature extraction
            # based on your DataProcessor class
            feature_vector = np.random.rand(260)  # Placeholder
            feature_vectors.append(feature_vector)
        
        return np.array(feature_vectors)
    except Exception as e:
        print(f"Error loading pose data: {e}")
        return None

def prepare_sequences(data, seq_len=150):
    """
    Prepare data sequences for model input.
    
    Args:
        data (np.array): Input data of shape (num_frames, features)
        seq_len (int): Length of each sequence
        
    Returns:
        torch.Tensor: Sequences of shape (num_sequences, seq_len, features)
    """
    num_frames, features = data.shape
    num_sequences = max(1, num_frames // seq_len)
    
    # Truncate or pad data to fit sequences
    data = data[:num_sequences * seq_len]
    sequences = data.reshape(num_sequences, seq_len, features)
    
    return torch.tensor(sequences, dtype=torch.float32)

def run_inference(model_path, data_path, seq_len=150):
    """
    Run inference on pose data using the trained model.
    
    Args:
        model_path (str): Path to the trained model
        data_path (str): Path to the pose data .npz file
        seq_len (int): Length of sequences for inference
    """
    # Load the trained model
    try:
        model = load_model(model_path)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load pose data
    pose_data = load_pose_data(data_path)
    if pose_data is None:
        print("Failed to load pose data")
        return
    
    print(f"Loaded pose data with shape: {pose_data.shape}")
    
    # Prepare sequences
    sequences = prepare_sequences(pose_data, seq_len)
    print(f"Prepared {sequences.shape[0]} sequences of length {seq_len}")
    
    # Run inference
    with torch.no_grad():
        predictions = model(sequences)
    
    # Process predictions
    print(f"Predictions shape: {predictions.shape}")
    print("Sample predictions (first sequence, first 10 frames):")
    print(predictions[0, :10, 0].numpy())
    
    # Convert to binary predictions (threshold = 0.5)
    binary_predictions = (predictions > 0.5).float()
    print("Binary predictions (first sequence, first 10 frames):")
    print(binary_predictions[0, :10, 0].int().numpy())
    
    return predictions

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with TennisPointDetector model")
    parser.add_argument("model_path", help="Path to the trained model file")
    parser.add_argument("data_path", help="Path to the pose data .npz file")
    parser.add_argument("--seq_len", type=int, default=150, help="Sequence length (default: 150)")
    
    args = parser.parse_args()
    
    run_inference(args.model_path, args.data_path, args.seq_len)