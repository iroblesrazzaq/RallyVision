import torch
from tennis_dataset import TennisDataset

# Create datasets for each split
train_dataset = TennisDataset('data/train.h5')
val_dataset = TennisDataset('data/val.h5')
test_dataset = TennisDataset('data/test.h5')

# Create data loaders with customizable batch size
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print dataset information
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Test loading a batch
for features, targets in train_loader:
    print(f"Feature batch shape: {features.shape}")
    print(f"Target batch shape: {targets.shape}")
    print(f"Feature dtype: {features.dtype}")
    print(f"Target dtype: {targets.dtype}")
    print(f"Sample target values: {targets[0, :10]}")  # First 10 targets of first sequence
    break