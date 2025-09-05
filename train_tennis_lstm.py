#!/usr/bin/env python3
"""
Training script for tennis point detection LSTM model.
"""

import torch
import torch.nn as nn
import os
import argparse
from tennis_dataset import TennisDataset, create_data_loaders
from tennis_lstm_model import TennisPointLSTM, train_model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Tennis Point Detection LSTM')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing feature vector .npy files')
    parser.add_argument('--annotations_dir', type=str, required=True,
                        help='Directory containing annotation CSV files')
    parser.add_argument('--model_save_path', type=str, default='tennis_point_lstm.pth',
                        help='Path to save trained model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
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
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker processes for data loading')
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        return
    
    if not os.path.exists(args.annotations_dir):
        print(f"Error: Annotations directory {args.annotations_dir} does not exist")
        return
    
    print("Initializing data loaders...")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        annotations_dir=args.annotations_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        feature_vector_size=args.feature_vector_size,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    print("Creating model...")
    
    # Create model
    model = TennisPointLSTM(
        input_size=args.feature_vector_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=False
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )
    
    # Save model
    torch.save(model.state_dict(), args.model_save_path)
    print(f"Model saved to {args.model_save_path}")
    
    # Test on test set
    print("Testing on test set...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            labels = labels.squeeze()
            
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')


if __name__ == "__main__":
    main()