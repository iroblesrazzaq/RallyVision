# %%
#!/usr/bin/env python3
"""
Training script for Tennis Point Detection LSTM Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import argparse
import time
from datetime import datetime
import matplotlib.pyplot as plt

from tennis_dataset import TennisDataset
from lstm_model_arch import TennisPointLSTM

# %%
# define hyperparameters
epochs = 50
batch_size = 32
lr = 0.001
hidden_size = 128
num_layers = 2
dropout = 0.2
bidirectional = True
pos_weight = 4.0
early_stopping_patience = 10
early_stopping_threshold = None
checkpoint_dir = 'checkpoints'
log_dir = 'logs'

# %%
train_dataset = TennisDataset('data/train.h5')
val_dataset = TennisDataset('data/val.h5')
test_dataset = TennisDataset('data/test.h5')

print(f"  Training samples: {len(train_dataset)}")
print(f"  Validation samples: {len(val_dataset)}")
print(f"  Test samples: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
def train_model(model, train_loader, val_loader, test_loader, 
                num_epochs=50, learning_rate=0.001, batch_size=32,
                pos_weight=None, early_stopping_patience=10, early_stopping_threshold=None,
                checkpoint_dir='checkpoints', log_dir='logs'):
    """
    Train the tennis point detection model.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate for optimizer
        batch_size (int): Batch size
        pos_weight (float): Positive class weight for weighted loss
        early_stopping_patience (int): Number of epochs to wait before early stopping
        early_stopping_threshold (float): Validation loss threshold for early stopping (optional)
        checkpoint_dir (str): Directory to save model checkpoints
        log_dir (str): Directory to save logs
    """
    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Loss function and optimizer
    if pos_weight is not None and pos_weight > 0:
        # Use weighted BCEWithLogitsLoss
        pos_weight_tensor = torch.tensor([pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"Using weighted loss with positive weight: {pos_weight}")
    else:
        # Standard BCE loss
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        print("Using standard BCE loss")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    epochs_since_improvement = 0
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(sequences)
            
            # Reshape for loss calculation
            batch_size, seq_length, _ = outputs.shape
            outputs_reshaped = outputs.view(batch_size * seq_length, 1)
            labels_reshaped = labels.view(batch_size * seq_length, 1).float()
            
            # Calculate loss
            if pos_weight is not None and pos_weight > 0:
                # For weighted loss, we use raw logits
                loss = criterion(outputs_reshaped, labels_reshaped)
            else:
                # Standard BCELoss
                loss = criterion(outputs_reshaped, labels_reshaped)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            # Calculate accuracy using probabilities
            predictions = (torch.sigmoid(outputs_reshaped) > 0.5).float()
            train_correct += (predictions == labels_reshaped).sum().item()
            train_total += labels_reshaped.numel()
            
            if batch_idx % 10 == 0:
                print(f'  Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # Calculate average training loss and accuracy
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device, 
                                              pos_weight if pos_weight is not None and pos_weight > 0 else None, 
                                              weighted_accuracy=True)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Check if this is the best model
        is_best = val_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy = val_accuracy
            
        # Early stopping check
        # Check if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            
        # Check early stopping conditions
        should_stop = False
        
        # Patience-based early stopping
        print(f"Early stopping patience: {early_stopping_patience}")
        if early_stopping_patience > 0 and epochs_since_improvement >= early_stopping_patience:
            print(f"Early stopping: No improvement in validation loss for {early_stopping_patience} epochs")
            should_stop = True
            
        # Threshold-based early stopping
        if early_stopping_threshold is not None and val_loss <= early_stopping_threshold:
            print(f"Early stopping: Validation loss {val_loss:.4f} reached threshold {early_stopping_threshold:.4f}")
            should_stop = True
            
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch+1, avg_train_loss, val_loss, val_accuracy, 
                       checkpoint_dir, is_best)
        
        # Print epoch statistics
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_accuracy:.4f}, '
              f'Val Acc: {val_accuracy:.4f}, '
              f'Time: {epoch_time:.2f}s')
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_accuracy,
            'best_val_loss': best_val_loss
        }
        
        with open(os.path.join(log_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)
            
        # Early stopping
        if should_stop:
            print(f"Stopping training at epoch {epoch+1}")
            break
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device,
                                            pos_weight if pos_weight is not None and pos_weight > 0 else None,
                                            weighted_accuracy=True)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Save final test results
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }
    
    with open(os.path.join(log_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f)
    
    # Plot training curves
    plot_path = os.path.join(log_dir, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, plot_path)
    
    return history

# %%
def evaluate_model(model, data_loader, criterion, device, pos_weight=None, weighted_accuracy=True):
    """
    Evaluate the model on a given dataset.
    
    Args:
        model (nn.Module): The model to evaluate
        data_loader (DataLoader): DataLoader for the dataset
        criterion (nn.Module): Loss function
        device (torch.device): Device to use for computation
        pos_weight (float): Positive class weight for weighted loss (optional)
        weighted_accuracy (bool): Whether to weight in-point predictions 4x more than not-in-point
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_pos_correct = 0  # Track positive class correct predictions
    total_pos_samples = 0  # Track positive class samples
    
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            
            # Reshape for loss calculation
            batch_size, seq_length, _ = outputs.shape
            outputs_reshaped = outputs.view(batch_size * seq_length, 1)
            labels_reshaped = labels.view(batch_size * seq_length, 1).float()
            
            # Calculate loss
            if pos_weight is not None and pos_weight > 0:
                # For weighted evaluation, we need to use BCEWithLogitsLoss
                pos_weight_tensor = torch.tensor([pos_weight]).to(device)
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
                # We need raw logits for BCEWithLogitsLoss
                loss = loss_fn(outputs_reshaped, labels_reshaped)
            else:
                loss = criterion(outputs_reshaped, labels_reshaped)
            total_loss += loss.item()
            
            # Calculate accuracy using probabilities
            predictions = (torch.sigmoid(outputs_reshaped) > 0.5).float()
            correct_predictions = (predictions == labels_reshaped)
            
            # Track overall accuracy
            total_correct += correct_predictions.sum().item()
            total_samples += labels_reshaped.numel()
            
            # Track positive class accuracy (weighted 4x)
            pos_mask = (labels_reshaped == 1)
            if pos_mask.sum() > 0:
                total_pos_correct += correct_predictions[pos_mask].sum().item()
                total_pos_samples += pos_mask.sum().item()
    
    avg_loss = total_loss / len(data_loader)
    
    # Calculate weighted accuracy: 4x weight for positive class
    if weighted_accuracy and total_pos_samples > 0:
        pos_accuracy = total_pos_correct / total_pos_samples
        overall_accuracy = total_correct / total_samples
        # Weighted accuracy: 4 parts positive accuracy, 1 part negative accuracy
        weighted_accuracy = (4 * pos_accuracy + overall_accuracy) / 5
        accuracy = weighted_accuracy
    else:
        accuracy = total_correct / total_samples
    
    return avg_loss, accuracy

# %%
def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path=None):
    """
    Plot training and validation curves.
    
    Args:
        train_losses (list): Training losses over epochs
        val_losses (list): Validation losses over epochs
        train_accuracies (list): Training accuracies over epochs
        val_accuracies (list): Validation accuracies over epochs
        save_path (str): Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
    ax1.plot(epochs, val_losses, label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy curves
    ax2.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    ax2.plot(epochs, val_accuracies, label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
    
    plt.show()

# %%
def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_accuracy, checkpoint_dir, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): Model to save
        optimizer (optim.Optimizer): Optimizer to save
        epoch (int): Current epoch
        train_loss (float): Current training loss
        val_loss (float): Current validation loss
        val_accuracy (float): Current validation accuracy
        checkpoint_dir (str): Directory to save checkpoints
        is_best (bool): Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved to {best_path}")

# %%
model = TennisPointLSTM(
    input_size=360,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout,
    bidirectional=bidirectional,
    return_logits=(pos_weight is not None and pos_weight > 0)  #
)


# %%
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    num_epochs=epochs,
    learning_rate=lr,
    batch_size=batch_size,
    pos_weight=pos_weight if pos_weight > 0 else None,
    early_stopping_patience=early_stopping_patience,
    early_stopping_threshold=early_stopping_threshold,
    checkpoint_dir=checkpoint_dir,
    log_dir=log_dir
)

# %%
# Evaluate best model on test set
def evaluate_best_model():
    """
    Load the best model and evaluate it on the test set.
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = TennisPointLSTM(
        input_size=360,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        return_logits=(pos_weight is not None and pos_weight > 0)
    )
    
    # Load best model
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if not os.path.exists(best_model_path):
        print(f"Best model not found at {best_model_path}")
        return
    
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Loaded best model from {best_model_path}")
    
    # Loss function
    if pos_weight is not None and pos_weight > 0:
        # Use weighted BCEWithLogitsLoss
        pos_weight_tensor = torch.tensor([pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        # Standard BCE loss
        criterion = nn.BCELoss()
    
    # Evaluate on test set
    print("\nEvaluating best model on test set...")
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device,
                                            pos_weight if pos_weight is not None and pos_weight > 0 else None,
                                            weighted_accuracy=False)  # Standard accuracy
    
    _, test_weighted_accuracy = evaluate_model(model, test_loader, criterion, device,
                                            pos_weight if pos_weight is not None and pos_weight > 0 else None,
                                            weighted_accuracy=True)   # Weighted accuracy
    
    print(f"Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Standard Accuracy: {test_accuracy:.4f}")
    print(f"  Weighted Accuracy: {test_weighted_accuracy:.4f}")
    
    # Save results
    test_results = {
        'test_loss': test_loss,
        'standard_accuracy': test_accuracy,
        'weighted_accuracy': test_weighted_accuracy
    }
    
    results_path = os.path.join(log_dir, 'best_model_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f)
    print(f"Test results saved to {results_path}")

# Run evaluation
evaluate_best_model()