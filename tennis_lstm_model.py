#!/usr/bin/env python3
"""
Simple LSTM model for tennis point detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TennisPointLSTM(nn.Module):
    """
    LSTM model for tennis point detection.
    Takes sequences of feature vectors and predicts whether each frame is during a point.
    """
    
    def __init__(self, input_size=288, hidden_size=128, num_layers=2, 
                 dropout=0.2, bidirectional=False):
        """
        Initialize the TennisPointLSTM.
        
        Args:
            input_size (int): Size of input feature vectors
            hidden_size (int): Number of features in the hidden state
            num_layers (int): Number of recurrent layers
            dropout (float): Dropout probability
            bidirectional (bool): If True, becomes a bidirectional LSTM
        """
        super(TennisPointLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output size from LSTM
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Binary classification: point/no point
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output for classification
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        last_output = lstm_out[:, -1, :]  # Take the last time step
        
        # Fully connected layers
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


# Alternative model that uses attention mechanism
class TennisPointLSTMWithAttention(nn.Module):
    """
    LSTM model with attention mechanism for tennis point detection.
    """
    
    def __init__(self, input_size=288, hidden_size=128, num_layers=2, 
                 dropout=0.2, bidirectional=False):
        """
        Initialize the TennisPointLSTMWithAttention.
        """
        super(TennisPointLSTMWithAttention, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output size from LSTM
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism
        self.attention = nn.Linear(lstm_output_size, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        """
        Forward pass with attention.
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        attention_weights = self.attention(lstm_out)
        attention_weights = self.softmax(attention_weights)
        
        # Apply attention weights
        weighted_output = lstm_out * attention_weights
        context_vector = torch.sum(weighted_output, dim=1)
        
        # Fully connected layers
        x = self.fc1(context_vector)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


# Training function example
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """
    Train the model.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate for optimizer
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)
            labels = labels.squeeze()  # Remove extra dimension
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                labels = labels.squeeze()
                
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_loss = train_loss / len(train_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Loss: {avg_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Acc: {val_acc:.2f}%')


# Example usage
if __name__ == "__main__":
    # Create model
    model = TennisPointLSTM(
        input_size=288,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    
    print("TennisPointLSTM Model Architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test with dummy input
    dummy_input = torch.randn(4, 150, 288)  # batch_size=4, seq_len=150, features=288
    output = model(dummy_input)
    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Create attention model
    attention_model = TennisPointLSTMWithAttention(
        input_size=288,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    
    print("\n\nTennisPointLSTMWithAttention Model Architecture:")
    print(attention_model)
    print(f"Total parameters: {sum(p.numel() for p in attention_model.parameters())}")
    
    # Test with dummy input
    attention_output = attention_model(dummy_input)
    print(f"\nAttention model dummy input shape: {dummy_input.shape}")
    print(f"Attention model output shape: {attention_output.shape}")