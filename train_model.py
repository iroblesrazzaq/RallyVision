#!/usr/bin/env python3
"""
train_model.py - Simple training script for the TennisPointDetector model
"""

import torch
import torch.nn as nn
import numpy as np
import os
from model import TennisPointDetector, save_model

def generate_sample_data(batch_size=32, seq_len=150, input_size=260):
    """
    Generate sample training data.
    
    Args:
        batch_size (int): Number of sequences in batch
        seq_len (int): Length of each sequence
        input_size (int): Size of input features
        
    Returns:
        tuple: (input_data, labels)
    """
    # Generate random input data
    X = torch.randn(batch_size, seq_len, input_size)
    
    # Generate random labels (0 or 1 for each frame)
    y = torch.randint(0, 2, (batch_size, seq_len, 1)).float()
    
    return X, y

def train_model():
    """
    Train the TennisPointDetector model on sample data.
    """
    # Model parameters
    input_size = 260
    hidden_size = 128
    num_layers = 2
    dropout = 0.2
    learning_rate = 0.001
    num_epochs = 10
    
    # Create model
    model = TennisPointDetector(input_size, hidden_size, num_layers, dropout)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting training...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    for epoch in range(num_epochs):
        # Generate sample data
        X, y = generate_sample_data()
        
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Save the trained model
    model_path = "tennis_point_detector.pth"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train the model
    trained_model = train_model()
    
    print("Training completed successfully!")