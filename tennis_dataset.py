#!/usr/bin/env python3
"""
Tennis Dataset for LSTM training.
Handles loading feature vectors and creating ground truth labels from annotations.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import glob


class TennisDataset(Dataset):
    """
    PyTorch Dataset for tennis point detection LSTM.
    Loads feature vectors and creates ground truth labels from annotations.
    """
    
    def __init__(self, data_dir, sequence_length=150, 
                 feature_vector_size=288, normalize=True, transform=None):
        """
        Initialize the TennisDataset.
        
        Args:
            data_dir (str): Directory containing feature vector .npy files and status files
            sequence_length (int): Length of sequences for LSTM (default 150 frames)
            feature_vector_size (int): Size of each feature vector (default 288)
            normalize (bool): Whether to normalize features
            transform (callable): Optional transform to be applied on samples
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.feature_vector_size = feature_vector_size
        self.normalize = normalize
        self.transform = transform
        
        # Find all feature files
        self.feature_files = sorted(glob.glob(os.path.join(data_dir, "*_features.npy")))
        
        # Load and process all data
        self.sequences, self.labels = self._load_all_data()
        
        # Normalize features if requested
        if self.normalize:
            self.scaler = StandardScaler()
            # Reshape to 2D for scaler, then back to 3D
            original_shape = self.sequences.shape
            self.sequences = self.sequences.reshape(-1, self.feature_vector_size)
            self.sequences = self.scaler.fit_transform(self.sequences)
            self.sequences = self.sequences.reshape(original_shape)
        
        print(f"Loaded {len(self.sequences)} sequences with {len(self.labels)} labels")
    
    def _load_all_data(self):
        """Load all feature vectors and create sequences with labels."""
        all_sequences = []
        all_labels = []
        
        for feature_file in self.feature_files:
            try:
                # Load feature vectors
                features = np.load(feature_file)
                print(f"Loaded features from {feature_file}: shape {features.shape}")
                
                # Load corresponding annotation status
                base_name = os.path.basename(feature_file)
                status_file = os.path.join(self.data_dir, base_name.replace("_features.npy", "_status.npy"))
                if os.path.exists(status_file):
                    annotation_status = np.load(status_file)
                    print(f"Loaded annotation status from {status_file}: shape {annotation_status.shape}")
                else:
                    # Create default annotation status if no file found
                    annotation_status = np.zeros(len(features))  # Default to 0 (not in play)
                    print(f"No annotation status file found for {feature_file}, using default status")
                
                # Create sequences and labels
                sequences, labels = self._create_sequences_and_labels(features, annotation_status)
                all_sequences.extend(sequences)
                all_labels.extend(labels)
                
            except Exception as e:
                print(f"Error loading {feature_file}: {e}")
                continue
        
        # Convert to numpy arrays
        if all_sequences:
            sequences_array = np.array(all_sequences)
            labels_array = np.array(all_labels)
            return sequences_array, labels_array
        else:
            # Return empty arrays with correct shapes
            return np.empty((0, self.sequence_length, self.feature_vector_size)), np.empty((0,))
    
    def _create_sequences_and_labels(self, features, annotation_status):
        """
        Create sequences and corresponding labels from features and annotation status.
        
        Args:
            features (np.array): Feature vectors of shape (num_frames, feature_vector_size)
            annotation_status (np.array): Annotation status for each frame
            
        Returns:
            tuple: (sequences, labels) lists
        """
        sequences = []
        labels = []
        
        num_frames = len(features)
        
        # Create sliding windows
        for i in range(0, max(1, num_frames - self.sequence_length + 1)):
            # Extract sequence
            end_idx = min(i + self.sequence_length, num_frames)
            sequence = features[i:end_idx]
            
            # Pad sequence if necessary
            if len(sequence) < self.sequence_length:
                padding = np.full((self.sequence_length - len(sequence), self.feature_vector_size), -1.0)
                sequence = np.vstack([sequence, padding])
            
            sequences.append(sequence)
            
            # Determine label for this sequence (middle frame)
            middle_frame = i + self.sequence_length // 2
            if middle_frame < len(annotation_status):
                # Use the annotation status as the label (0 for not in play, 1 for in play)
                # Convert any negative values to 0
                label = max(0, annotation_status[middle_frame])
            else:
                label = 0  # Default to not in play
                
            labels.append(label)
        
        return sequences, labels
    
    def __len__(self):
        """Return the number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a single sequence and its label."""
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert to PyTorch tensors
        sequence_tensor = torch.FloatTensor(sequence)
        label_tensor = torch.LongTensor([label])
        
        if self.transform:
            sequence_tensor = self.transform(sequence_tensor)
            
        return sequence_tensor, label_tensor


def create_data_loaders(data_dir, batch_size=32, 
                       sequence_length=150, feature_vector_size=288,
                       train_split=0.8, val_split=0.1, test_split=0.1,
                       num_workers=0, shuffle=True):
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir (str): Directory containing feature vector .npy files and status files
        batch_size (int): Batch size for data loaders
        sequence_length (int): Length of sequences for LSTM
        feature_vector_size (int): Size of each feature vector
        train_split (float): Proportion of data for training
        val_split (float): Proportion of data for validation
        test_split (float): Proportion of data for testing
        num_workers (int): Number of worker processes
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create dataset
    dataset = TennisDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        feature_vector_size=feature_vector_size
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    print(f"Data loaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Validation: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Example of how to use the TennisDataset
    print("TennisDataset - Example Usage")
    print("=" * 40)
    
    # Initialize dataset
    dataset = TennisDataset(
        data_dir="/path/to/feature/vectors"
    )
    
    # Get a sample
    if len(dataset) > 0:
        sample_sequence, sample_label = dataset[0]
        print(f"Sample sequence shape: {sample_sequence.shape}")
        print(f"Sample label: {sample_label}")
    
    # Create data loaders
    # train_loader, val_loader, test_loader = create_data_loaders(
    #     data_dir="/path/to/feature/vectors"
    # )