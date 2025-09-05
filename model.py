import torch
import torch.nn as nn
import numpy as np

class TennisPointDetector(nn.Module):
    """
    Bidirectional LSTM model for detecting tennis points from pose data.
    
    Takes sequence of pose features and predicts whether each frame is during a point or not.
    """
    
    def __init__(self, input_size=260, hidden_size=128, num_layers=2, dropout=0.2):
        """
        Initialize the TennisPointDetector model.
        
        Args:
            input_size (int): Size of input features (260 for pose data)
            hidden_size (int): Number of features in the hidden state
            num_layers (int): Number of recurrent layers
            dropout (float): Dropout rate for regularization
        """
        super(TennisPointDetector, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer to map to output
        # Bidirectional doubles the hidden size
        self.fc = nn.Linear(hidden_size * 2, 1)  # Single output (point/not point)
        
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, 1)
        """
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Pass through fully connected layer
        out = self.fc(lstm_out)
        
        # Apply sigmoid activation
        out = self.sigmoid(out)
        
        return out

def create_model(input_size=260, hidden_size=128, num_layers=2, dropout=0.2):
    """
    Create and return a TennisPointDetector model.
    
    Args:
        input_size (int): Size of input features
        hidden_size (int): Number of features in the hidden state
        num_layers (int): Number of recurrent layers
        dropout (float): Dropout rate for regularization
        
    Returns:
        TennisPointDetector: Initialized model
    """
    model = TennisPointDetector(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    return model

def load_model(model_path, input_size=260, hidden_size=128, num_layers=2, dropout=0.2):
    """
    Load a saved TennisPointDetector model.
    
    Args:
        model_path (str): Path to the saved model
        input_size (int): Size of input features
        hidden_size (int): Number of features in the hidden state
        num_layers (int): Number of recurrent layers
        dropout (float): Dropout rate for regularization
        
    Returns:
        TennisPointDetector: Loaded model
    """
    model = create_model(input_size, hidden_size, num_layers, dropout)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def save_model(model, model_path):
    """
    Save a TennisPointDetector model.
    
    Args:
        model (TennisPointDetector): Model to save
        model_path (str): Path to save the model
    """
    torch.save(model.state_dict(), model_path)

# Example usage:
if __name__ == "__main__":
    # Create model
    model = create_model()
    
    # Print model architecture
    print("Tennis Point Detector Model Architecture:")
    print("=" * 50)
    print(model)
    print("=" * 50)
    
    # Test with sample input
    batch_size = 4
    seq_len = 150  # 10 seconds at 15 FPS
    input_size = 260
    
    # Create sample input
    sample_input = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    with torch.no_grad():
        output = model(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("Model created successfully!")