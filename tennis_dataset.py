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
    
    def __init__(self, data_dir, annotations_dir, sequence_length=150, 
                 feature_vector_size=288, normalize=True, transform=None):
        """
        Initialize the TennisDataset.
        
        Args:
            data_dir (str): Directory containing feature vector .npy files
            annotations_dir (str): Directory containing annotation CSV files
            sequence_length (int): Length of sequences for LSTM (default 150 frames)
            feature_vector_size (int): Size of each feature vector (default 288)
            normalize (bool): Whether to normalize features
            transform (callable): Optional transform to be applied on samples
        """
        self.data_dir = data_dir
        self.annotations_dir = annotations_dir
        self.sequence_length = sequence_length
        self.feature_vector_size = feature_vector_size
        self.normalize = normalize
        self.transform = transform
        
        # Find all feature files and their corresponding annotation files
        self.feature_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        self.annotation_files = self._find_annotation_files()
        
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
    
    def _find_annotation_files(self):
        """Find corresponding annotation files for feature files."""
        annotation_files = {}
        for feature_file in self.feature_files:
            # Extract video name from feature file
            base_name = os.path.basename(feature_file)
            video_name = base_name.replace("_features.npy", "")
            
            # Look for corresponding annotation file
            annotation_pattern = os.path.join(self.annotations_dir, f"{video_name}*.csv")
            matching_annotations = glob.glob(annotation_pattern)
            
            if matching_annotations:
                annotation_files[feature_file] = matching_annotations[0]
            else:
                print(f"Warning: No annotation file found for {feature_file}")
                annotation_files[feature_file] = None
                
        return annotation_files
    
    def _load_all_data(self):
        """Load all feature vectors and create sequences with labels."""
        all_sequences = []
        all_labels = []
        
        for feature_file in self.feature_files:
            try:
                # Load feature vectors
                features = np.load(feature_file)
                print(f"Loaded features from {feature_file}: shape {features.shape}")
                
                # Load corresponding annotations
                annotation_file = self.annotation_files[feature_file]
                if annotation_file and os.path.exists(annotation_file):
                    annotations = pd.read_csv(annotation_file)
                    print(f"Loaded annotations from {annotation_file}: shape {annotations.shape}")
                else:
                    # Create empty annotations if no file found
                    annotations = pd.DataFrame(columns=['start_frame', 'end_frame', 'label'])
                    print(f"No annotations found for {feature_file}, using empty annotations")
                
                # Create sequences and labels
                sequences, labels = self._create_sequences_and_labels(features, annotations)
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
    
    def _create_sequences_and_labels(self, features, annotations):
        """
        Create sequences and corresponding labels from features and annotations.
        
        Args:
            features (np.array): Feature vectors of shape (num_frames, feature_vector_size)
            annotations (pd.DataFrame): Annotation data with start_frame, end_frame, label columns
            
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
            label = self._get_label_for_frame(middle_frame, annotations)
            labels.append(label)
        
        return sequences, labels
    
    def _get_label_for_frame(self, frame_idx, annotations):
        """
        Get label for a specific frame from annotations.
        
        Args:
            frame_idx (int): Frame index
            annotations (pd.DataFrame): Annotation data
            
        Returns:
            int: Label (1 for point/play, 0 for no point)
        """
        # Default label is 0 (no point)
        label = 0
        
        # Check if frame is within any annotated point segment
        for _, row in annotations.iterrows():
            start_frame = int(row['start_frame']) if 'start_frame' in row else 0
            end_frame = int(row['end_frame']) if 'end_frame' in row else 0
            
            if start_frame <= frame_idx <= end_frame:
                label = 1
                break
                
        return label
    
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


def create_data_loaders(data_dir, annotations_dir, batch_size=32, 
                       sequence_length=150, feature_vector_size=288,
                       train_split=0.8, val_split=0.1, test_split=0.1,
                       num_workers=0, shuffle=True):
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir (str): Directory containing feature vector .npy files
        annotations_dir (str): Directory containing annotation CSV files
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
        annotations_dir=annotations_dir,
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
        data_dir="/path/to/feature/vectors",
        annotations_dir="/path/to/annotations"
    )
    
    # Get a sample
    if len(dataset) > 0:
        sample_sequence, sample_label = dataset[0]
        print(f"Sample sequence shape: {sample_sequence.shape}")
        print(f"Sample label: {sample_label}")
    
    # Create data loaders
    # train_loader, val_loader, test_loader = create_data_loaders(
    #     data_dir="/path/to/feature/vectors",
    #     annotations_dir="/path/to/annotations"
    # )