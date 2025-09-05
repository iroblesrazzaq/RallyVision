#!/usr/bin/env python3
"""
evaluate.py - Evaluation script for the TennisPointDetector model
"""

import torch
import torch.nn as nn
import numpy as np
from model import TennisPointDetector

def generate_test_data(num_samples=100, seq_len=150, input_size=260):
    """
    Generate test data with known labels for evaluation.
    
    Args:
        num_samples (int): Number of samples to generate
        seq_len (int): Length of each sequence
        input_size (int): Size of input features
        
    Returns:
        tuple: (input_data, labels)
    """
    # Generate random input data
    X = torch.randn(num_samples, seq_len, input_size)
    
    # Generate labels with some pattern (for demonstration)
    # In reality, you would load actual labels
    y = torch.randint(0, 2, (num_samples, seq_len, 1)).float()
    
    return X, y

def evaluate_model(model, test_data, test_labels, criterion):
    """
    Evaluate the model on test data.
    
    Args:
        model (TennisPointDetector): Trained model
        test_data (torch.Tensor): Test input data
        test_labels (torch.Tensor): Test labels
        criterion (nn.Module): Loss function
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        loss = criterion(outputs, test_labels)
        
        # Calculate accuracy
        predicted = (outputs > 0.5).float()
        correct = (predicted == test_labels).sum().item()
        total = test_labels.numel()
        accuracy = correct / total
        
    return loss.item(), accuracy

def main():
    # Model parameters
    input_size = 260
    hidden_size = 128
    num_layers = 2
    dropout = 0.2
    
    # Create model
    model = TennisPointDetector(input_size, hidden_size, num_layers, dropout)
    
    # Load a pre-trained model if available
    try:
        model.load_state_dict(torch.load("tennis_point_detector.pth"))
        print("Loaded pre-trained model")
    except FileNotFoundError:
        print("No pre-trained model found, using untrained model for demonstration")
    
    # Generate test data
    test_data, test_labels = generate_test_data(num_samples=50)
    print(f"Generated test data with shape: {test_data.shape}")
    
    # Define loss function
    criterion = nn.BCELoss()
    
    # Evaluate model
    avg_loss, accuracy = evaluate_model(model, test_data, test_labels, criterion)
    
    print(f"Evaluation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Show some sample predictions
    model.eval()
    with torch.no_grad():
        sample_outputs = model(test_data[:3])  # First 3 samples
        sample_predictions = (sample_outputs > 0.5).float()
        
        print("\nSample Predictions (first 5 frames of first 3 sequences):")
        for i in range(3):
            print(f"Sequence {i+1}:")
            print(f"  Predictions: {sample_predictions[i, :5, 0].int().numpy()}")
            print(f"  Actual:      {test_labels[i, :5, 0].int().numpy()}")

if __name__ == "__main__":
    main()