"""Bidirectional LSTM for temporal intent prediction."""

import torch
import torch.nn as nn


class BidirectionalLSTMIntentClassifier(nn.Module):
    """Bidirectional LSTM that explicitly models temporal sequences.
    
    Key advantage over transformer: BiLSTM naturally captures sequential dependencies
    without attention overhead, making it better for temporal understanding.
    
    - Input: (B, T, pose_dim) pose sequences with velocity
    - Output: (B, 1) sigmoid probabilities for binary classification
    - Uses BCELoss (same as other models)
    """

    def __init__(
        self,
        input_size: int = 68,  # pose (34) + velocity (34)
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Bidirectional LSTM: processes sequence forward AND backward
        # This helps the model understand temporal context better
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,  # KEY: bidirectional processing
        )

        # After BiLSTM, we have hidden_size*2 dimensions (forward + backward)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_size) pose sequence with velocity
        Returns:
            (B, 1) intent probability
        """
        # BiLSTM returns (output, (hidden, cell))
        # output shape: (B, T, hidden_size*2) - concatenation of forward and backward
        lstm_out, (hidden, cell) = self.bilstm(x)
        
        # Take the last output frame from BiLSTM
        # This represents the full temporal context up to the current moment
        last_output = lstm_out[:, -1, :]  # (B, hidden_size*2)
        
        if self.dropout is not None:
            last_output = self.dropout(last_output)
        
        # Classification
        out = self.fc(last_output)
        return out


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = BidirectionalLSTMIntentClassifier(input_size=68)
    B, T = 4, 25
    pose_with_vel = torch.randn(B, T, 68)  # 34 pose + 34 velocity
    output = model(pose_with_vel)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print("✓ Bidirectional LSTM for temporal intent prediction")
