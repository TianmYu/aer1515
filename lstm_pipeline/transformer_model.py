"""Improved transformer for intent prediction from 3D pose sequences."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoding: torch.Tensor
        self.register_buffer("pos_encoding", pe.unsqueeze(0))  # (1, max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.pos_encoding.shape[1]:
            raise ValueError("Sequence length exceeds maximum supported by positional encoding")
        x = x + self.pos_encoding[:, : x.shape[1]]
        return self.dropout(x)


class SimpleTransformerIntentClassifier(nn.Module):
    """Temporal transformer that uses pose motion cues and padding masks.
    
    Compatible with LSTM training pipeline:
    - Input: (B, T, pose_dim) pose sequences
    - Output: (B, 1) logits for binary classification (use BCEWithLogitsLoss)
    - Returns logits (no final sigmoid) to allow stable pos-weighting with BCEWithLogitsLoss
    """

    def __init__(
        self,
        input_size: int = 68,  # Renamed from pose_dim for LSTM compatibility
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 4,
        downsample_layers: int = 2,
        use_velocity_features: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.pose_dim = input_size  # Alias for internal use
        self.d_model = d_model
        self.use_velocity_features = use_velocity_features

        # Input dimension: pose + velocity only (simpler and more stable)
        frame_input_dim = input_size * (2 if use_velocity_features else 1)

        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(frame_input_dim),
            nn.Linear(frame_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.downsample_layers = nn.ModuleList()
        for _ in range(max(0, downsample_layers)):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Causal masking can be enabled for inference, but often hurts training
        self.use_causal_mask = False  # Disabled by default - bidirectional context helps learning

        # Binary classifier that returns logits. Use dropout before final linear layer.
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),  # logits output
        )

    def _build_velocity(self, pose: torch.Tensor) -> torch.Tensor:
        velocity = torch.zeros_like(pose)
        if pose.size(1) > 1:
            velocity[:, 1:] = pose[:, 1:] - pose[:, :-1]
        return velocity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, pose_dim) 3D pose sequence (zero-padded)
        Returns:
            (B, 1) intent probability [0, 1] via sigmoid
        """
        pose = x  # Rename for compatibility with LSTM interface
        B, T, _ = pose.shape
        device = pose.device

        # Automatically detect valid frames from non-zero content
        valid_frames = (pose.abs().sum(dim=-1) > 0)

        if self.use_velocity_features:
            velocity = self._build_velocity(pose)
            frame_feats = torch.cat([pose, velocity], dim=-1)
        else:
            frame_feats = pose

        x = self.frame_encoder(frame_feats)  # (B, T, d_model)

        if len(self.downsample_layers) > 0:
            x = x.transpose(1, 2)  # (B, d_model, T)
            mask = valid_frames.float().unsqueeze(1)
            for layer in self.downsample_layers:
                x = layer(x)
                mask = F.max_pool1d(mask, kernel_size=2, stride=2, ceil_mode=True)
            x = x.transpose(1, 2)
            valid_frames = mask.squeeze(1) > 0.5

        x = self.pos_encoder(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        padding_mask = torch.cat([cls_mask, ~valid_frames], dim=1)

        # Create causal attention mask (prevent looking into future)
        attn_mask = None
        if self.use_causal_mask:
            seq_len = x.shape[1]
            attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        encoded = self.transformer(x, mask=attn_mask, src_key_padding_mask=padding_mask)
        cls_out = encoded[:, 0, :]
        # return logits (no sigmoid here). Apply torch.sigmoid at inference/time-of-use.
        return self.classifier(cls_out)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SimpleTransformerIntentClassifier(input_size=68)
    B, T = 4, 120
    pose = torch.randn(B, T, 68)
    output = model(pose)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print("✓ Compatible with LSTM training pipeline (BCELoss)")
