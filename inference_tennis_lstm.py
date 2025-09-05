#!/usr/bin/env python3
"""
Inference script for tennis point detection LSTM model.
"""

import torch
import numpy as np
import os
import argparse
from tennis_lstm_model import TennisPointLSTM


def load_model(model_path, input_size=288, hidden_size=128, num_layers=2, dropout=0.2):
    """
    Load trained model from file.
    
    Args:
        model_path (str): Path to saved model
        input_size (int): Size of input feature vectors
        hidden_size (int): Hidden size for LSTM
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
        
    Returns:
        TennisPointLSTM: Loaded model
    """
    model = TennisPointLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model


def predict_sequence(model, sequence, device):
    """
    Predict whether a sequence contains a point.
    
    Args:
        model (TennisPointLSTM): Trained model
        sequence (np.array): Sequence of feature vectors
        device (torch.device): Device to run inference on
        
    Returns:
        tuple: (prediction, confidence) - prediction (0/1), confidence (float)
    """
    # Convert to tensor
    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
    sequence_tensor = sequence_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(sequence_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        
    return prediction.item(), confidence.item()


def process_video_features(model, features_file, sequence_length=150):
    """
    Process all sequences in a video features file.
    
    Args:
        model (TennisPointLSTM): Trained model
        features_file (str): Path to features .npy file
        sequence_length (int): Length of sequences
        
    Returns:
        list: List of (frame_index, prediction, confidence) tuples
    """
    # Load features
    features = np.load(features_file)
    print(f"Loaded features: {features.shape}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    results = []
    
    # Process sequences
    num_frames = len(features)
    for i in range(0, max(1, num_frames - sequence_length + 1)):
        # Extract sequence
        end_idx = min(i + sequence_length, num_frames)
        sequence = features[i:end_idx]
        
        # Pad sequence if necessary
        if len(sequence) < sequence_length:
            padding = np.full((sequence_length - len(sequence), features.shape[1]), -1.0)
            sequence = np.vstack([sequence, padding])
        
        # Get prediction
        prediction, confidence = predict_sequence(model, sequence, device)
        
        # Store result (use middle frame as representative)
        middle_frame = i + sequence_length // 2
        results.append((middle_frame, prediction, confidence))
        
        if i % 50 == 0:
            print(f"Processed sequence starting at frame {i}")
    
    return results


def save_predictions(results, output_file):
    """
    Save predictions to CSV file.
    
    Args:
        results (list): List of (frame_index, prediction, confidence) tuples
        output_file (str): Path to output CSV file
    """
    import pandas as pd
    
    df = pd.DataFrame(results, columns=['frame_index', 'prediction', 'confidence'])
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run inference with trained Tennis Point Detection LSTM')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--features_file', type=str, required=True,
                        help='Path to features .npy file')
    parser.add_argument('--output_file', type=str, default='predictions.csv',
                        help='Path to output CSV file')
    parser.add_argument('--sequence_length', type=int, default=150,
                        help='Sequence length for LSTM')
    parser.add_argument('--feature_vector_size', type=int, default=288,
                        help='Size of feature vectors')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden size for LSTM')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist")
        return
    
    if not os.path.exists(args.features_file):
        print(f"Error: Features file {args.features_file} does not exist")
        return
    
    print("Loading model...")
    model = load_model(
        model_path=args.model_path,
        input_size=args.feature_vector_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    print("Processing video features...")
    results = process_video_features(
        model=model,
        features_file=args.features_file,
        sequence_length=args.sequence_length
    )
    
    print(f"Processed {len(results)} sequences")
    
    # Count predictions
    point_count = sum(1 for _, pred, _ in results if pred == 1)
    no_point_count = sum(1 for _, pred, _ in results if pred == 0)
    print(f"Points detected: {point_count}")
    print(f"No points detected: {no_point_count}")
    
    # Save results
    save_predictions(results, args.output_file)
    
    # Show some sample predictions
    print("\nSample predictions:")
    for i, (frame_idx, pred, conf) in enumerate(results[:10]):
        status = "POINT" if pred == 1 else "NO POINT"
        print(f"  Frame {frame_idx}: {status} (confidence: {conf:.3f})")


if __name__ == "__main__":
    main()
