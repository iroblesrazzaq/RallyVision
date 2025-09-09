import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class TennisDatasetLazy(Dataset):
    """
    A memory-efficient PyTorch Dataset class for loading tennis pose data from HDF5 files.
    Loads data on-demand rather than pre-loading everything into memory.
    """
    
    def __init__(self, hdf5_file_path):
        """
        Initialize the TennisDatasetLazy.
        
        Args:
            hdf5_file_path (str): Path to the HDF5 file containing the dataset
        """
        self.hdf5_file_path = hdf5_file_path
        
        # Open the HDF5 file and store references (not data)
        with h5py.File(self.hdf5_file_path, 'r') as f:
            self.features_shape = f['features'].shape
            self.targets_shape = f['targets'].shape
        
    def __len__(self):
        """
        Return the number of sequences in the dataset.
        
        Returns:
            int: Number of sequences
        """
        return self.features_shape[0]
    
    def __getitem__(self, idx):
        """
        Load and return a single sequence and its corresponding targets.
        
        Args:
            idx (int): Index of the sequence to retrieve
            
        Returns:
            tuple: (features, targets) where features is a tensor of shape (150, 360)
                   and targets is a tensor of shape (150,)
        """
        with h5py.File(self.hdf5_file_path, 'r') as f:
            features = f['features'][idx]
            targets = f['targets'][idx]
            
        # Convert to PyTorch tensors
        features = torch.FloatTensor(features)
        targets = torch.LongTensor(targets)
        
        return features, targets

# Example usage:
# dataset = TennisDatasetLazy('data/train.h5')
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)