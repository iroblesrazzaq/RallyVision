import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class TennisDataset(Dataset):
    """
    A PyTorch Dataset class for loading tennis pose data from HDF5 files.
    Different from the data pipeline side dataset creator which saves our data sequences to h5 format, 
    this dataset is compatible with pytorch api so we can just use pytorch dataloader, etc. 
    
    The dataset contains sequences of pose data with corresponding labels indicating
    whether each frame is part of a tennis point (1) or not (0).
    """
    
    def __init__(self, hdf5_file_path):
        """
        Initialize the TennisDataset.
        
        Args:
            hdf5_file_path (str): Path to the HDF5 file containing the dataset
        """
        self.hdf5_file_path = hdf5_file_path
        
        # Open the HDF5 file and load the datasets
        with h5py.File(self.hdf5_file_path, 'r') as f:
            self.features = f['features'][:]
            self.targets = f['targets'][:]
        
        # Convert to PyTorch tensors
        self.features = torch.FloatTensor(self.features)
        self.targets = torch.LongTensor(self.targets)
        
    def __len__(self):
        """
        Return the number of sequences in the dataset.
        
        Returns:
            int: Number of sequences
        """
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        """
        Get a single sequence and its corresponding targets.
        
        Args:
            idx (int): Index of the sequence to retrieve
            
        Returns:
            tuple: (features, targets) where features is a tensor of shape (150, 360)
                   and targets is a tensor of shape (150,)
        """
        return self.features[idx], self.targets[idx]
