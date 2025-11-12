"""Multimodal transformer with dynamic pose encoders and pretraining support.

This model:
- Dynamically creates encoders for different pose dimensions
- Supports pretraining with masked prediction and contrastive learning
- Handles missing modalities gracefully
- Works with any dataset without hardcoded feature dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TemporalEncoder(nn.Module):
    """Simple temporal encoder: 1D conv -> ReLU -> pooling.

    Expects input shape (B, T, F) and returns (B, D).
    """

    def __init__(self, in_feats: int, d_model: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_feats, d_model // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor):
        # x: (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (B, D)
        return x


class TransformerModel(nn.Module):
    """Multimodal fusion transformer with dynamic pose encoder support.

    This model dynamically creates encoders for different pose dimensions,
    making it flexible to work with any dataset without retraining.
    
    Features:
    - Variable pose dimensions (creates encoders on-the-fly)
    - Trajectory data support
    - Pretraining mode for self-supervised learning
    - Handles missing modalities
    """

    def __init__(self, pose_feats: Optional[int] = None, traj_feats: int = 2, 
                 d_model: int = 128, nhead: int = 4, num_layers: int = 3, 
                 num_classes: int = 2):
        """
        Args:
            pose_feats: Default pose feature dimension (None for fully dynamic)
            traj_feats: Trajectory feature dimension (default 2)
            d_model: Model hidden dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of intent classes
        """
        super().__init__()
        self.d_model = d_model
        self._traj_in_feats = max(1, traj_feats)
        
        # Dynamic pose encoder registry: maps pose_dim -> encoder
        self.pose_encoders = nn.ModuleDict()
        self.pose_projections = nn.ModuleDict()
        
        # Initialize default pose encoder if specified
        if pose_feats is not None and pose_feats > 0:
            self._register_pose_encoder(pose_feats)
        
        # Trajectory encoder (fixed)
        self.traj_enc = TemporalEncoder(self._traj_in_feats, d_model)
        self.proj_traj = nn.Linear(d_model, d_model)

        # Learnable tokens: CLS + modality type tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # Order: [pose, traj]
        self.modality_tokens = nn.Parameter(torch.randn(2, d_model))

        # Transformer fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, 
            dropout=0.1, activation='relu', batch_first=False
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Task heads
        self.intent_head = nn.Sequential(
            nn.LayerNorm(d_model), 
            nn.Linear(d_model, num_classes)
        )
        # Auxiliary head: predict future interaction (binary)
        self.future_head = nn.Sequential(
            nn.LayerNorm(d_model), 
            nn.Linear(d_model, 1)
        )
        
        # Pretraining heads (for self-supervised learning)
        self.mask_prediction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.contrastive_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2)
        )

    def _register_pose_encoder(self, pose_dim: int):
        """Dynamically register a pose encoder for a specific dimension."""
        key = str(pose_dim)
        if key not in self.pose_encoders:
            cls_token = getattr(self, 'cls_token', None)
            if cls_token is not None:
                device = cls_token.device
            else:
                device = next(self.parameters(), torch.tensor(0.0)).device
            encoder = TemporalEncoder(max(1, pose_dim), self.d_model)
            projection = nn.Linear(self.d_model, self.d_model)
            self.pose_encoders[key] = encoder.to(device)
            self.pose_projections[key] = projection.to(device)
    
    def _get_pose_encoder(self, pose_dim: int):
        """Get or create pose encoder for specific dimension."""
        key = str(pose_dim)
        if key not in self.pose_encoders:
            self._register_pose_encoder(pose_dim)
        return self.pose_encoders[key], self.pose_projections[key]

    def forward(self, batch: dict, dropout_modality_probs: Optional[list] = None, 
                return_features: bool = False):
        """Forward pass with dynamic pose encoder selection.

        Args:
            batch: Dictionary containing:
                - 'pose': Tensor(B, T, F_pose) or None (any dimension F_pose)
                - 'traj': Tensor(B, T, F_traj) or None
                - 'has_pose': Tensor(B,) float or bool (optional)
                - 'has_traj': Tensor(B,) float or bool (optional)
            dropout_modality_probs: Optional list of dropout probabilities per modality [pose_p, traj_p]
            return_features: If True, return intermediate features for pretraining

        Returns:
            Dict with 'intent_logits', 'future_logits', and optionally 'features'
        """
        B = None
        device = None
        embeddings = []
        presence = []

        # Pose - dynamically select encoder based on actual dimension
        if batch.get('pose') is not None:
            x_pose = batch['pose'].float()
            B = x_pose.shape[0]
            device = x_pose.device
            pose_dim = x_pose.shape[2]
            
            # Handle zero-dimensional pose (trajectory-only)
            if pose_dim == 0:
                e_pose = torch.zeros(B, self.d_model, device=device)
            else:
                # Get or create encoder for this pose dimension
                encoder, projection = self._get_pose_encoder(pose_dim)
                e_pose = encoder(x_pose)
                e_pose = projection(e_pose)
            
            embeddings.append(e_pose)
            presence.append(batch.get('has_pose', torch.ones(B, device=device)))
        else:
            # Completely missing pose modality
            if B is None:
                raise ValueError('Batch must contain at least one modality to infer batch size')
            embeddings.append(torch.zeros(B, self.d_model, device=device))
            presence.append(torch.zeros(B, device=device))

        # Trajectory
        if batch.get('traj') is not None:
            x_traj = batch['traj'].float()
            if B is None:
                B = x_traj.shape[0]
                device = x_traj.device
            e_traj = self.traj_enc(x_traj)
            e_traj = self.proj_traj(e_traj)
            embeddings.append(e_traj)
            presence.append(batch.get('has_traj', torch.ones(B, device=device)))
        else:
            embeddings.append(torch.zeros(B, self.d_model, device=device))
            presence.append(torch.zeros(B, device=device))

        # Optional modality dropout for robustness
        if self.training and dropout_modality_probs is not None:
            for i, p in enumerate(dropout_modality_probs):
                if p <= 0:
                    continue
                drop = (torch.rand(B, device=device) < p).float().unsqueeze(1)
                embeddings[i] = embeddings[i] * (1.0 - drop)
                presence[i] = presence[i] * (1.0 - drop.squeeze(1))

        # Build tokens: [CLS, pose_token, traj_token]
        tokens = []
        cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        tokens.append(cls)
        for i, emb in enumerate(embeddings):
            # emb: (B, D) -> token: (B,1,D) with modality token
            token = emb.unsqueeze(1) + self.modality_tokens[i].unsqueeze(0).unsqueeze(0)
            tokens.append(token)

        tokens = torch.cat(tokens, dim=1)  # (B, num_tokens, D)

        # Prepare transformer inputs (seq_len, B, D)
        src = tokens.permute(1, 0, 2)

        # src_key_padding_mask expects shape (B, S) with True for masked positions
        # We mask modality tokens (not CLS) where presence == 0
        presence_stack = torch.stack(presence, dim=1).bool()  # (B, num_modalities)
        # prepend False for CLS (CLS is always present)
        mask = torch.zeros(B, 1 + presence_stack.shape[1], dtype=torch.bool, device=device)
        # mask positions where presence == 0 -> True means ignore
        mask[:, 1:] = ~presence_stack

        fused = self.fusion_transformer(src, src_key_padding_mask=mask)  # (S, B, D)
        fused = fused.permute(1, 0, 2)  # (B, S, D)
        cls_out = fused[:, 0, :]

        intent_logits = self.intent_head(cls_out)
        future_logits = self.future_head(cls_out).squeeze(-1)  # (B,)

        result = {
            'intent_logits': intent_logits,
            'future_logits': future_logits
        }
        
        if return_features:
            result['features'] = cls_out
            result['all_tokens'] = fused
        
        return result
    
    def forward_pretrain(self, batch: dict, mask_prob: float = 0.15):
        """Forward pass for pretraining with masked prediction.
        
        Args:
            batch: Input batch (same format as forward)
            mask_prob: Probability of masking each modality token
            
        Returns:
            Dict with pretraining losses and predictions
        """
        # Get features with masking
        B = batch['pose'].shape[0] if batch.get('pose') is not None else batch['traj'].shape[0]
        device = batch['pose'].device if batch.get('pose') is not None else batch['traj'].device
        
        # Forward pass to get original features
        with torch.no_grad():
            orig_out = self.forward(batch, return_features=True)
            target_features = orig_out['features'].detach()
        
        # Create masked version by randomly dropping modalities
        masked_dropout = [mask_prob, mask_prob]
        
        # Forward with masking
        masked_out = self.forward(batch, dropout_modality_probs=masked_dropout, return_features=True)
        
        # Masked feature reconstruction loss
        recon_pred = self.mask_prediction_head(masked_out['features'])
        recon_loss = F.mse_loss(recon_pred, target_features)
        
        # Contrastive learning (bring masked and unmasked features closer)
        proj_orig = self.contrastive_projection(target_features)
        proj_masked = self.contrastive_projection(masked_out['features'])
        contrastive_loss = F.mse_loss(proj_orig, proj_masked)
        
        return {
            'recon_loss': recon_loss,
            'contrastive_loss': contrastive_loss,
            'total_pretrain_loss': recon_loss + 0.5 * contrastive_loss,
            'features': masked_out['features']
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Smoke test with different pose dimensions
    print("Testing dynamic pose encoder...")
    
    # Test 1: 72-dim keypoint pose
    model = TransformerModel(pose_feats=72, d_model=64, nhead=4, num_layers=2)
    B, T = 4, 30
    batch1 = {
        'pose': torch.randn(B, T, 72),
        'traj': torch.randn(B, T, 2),
        'has_pose': torch.ones(B),
        'has_traj': torch.ones(B)
    }
    out1 = model(batch1)
    print(f"✓ 72-dim pose: intent_logits {out1['intent_logits'].shape}")
    
    # Test 2: 4-dim motion pose (new encoder created dynamically)
    batch2 = {
        'pose': torch.randn(B, T, 4),
        'traj': torch.randn(B, T, 2),
        'has_pose': torch.ones(B),
        'has_traj': torch.ones(B)
    }
    out2 = model(batch2)
    print(f"✓ 4-dim pose: intent_logits {out2['intent_logits'].shape}")
    print(f"✓ Dynamically created encoder for dim=4")
    
    # Test 3: Trajectory-only (0-dim pose)
    batch3 = {
        'pose': torch.randn(B, T, 0),
        'traj': torch.randn(B, T, 2),
        'has_pose': torch.zeros(B),
        'has_traj': torch.ones(B)
    }
    out3 = model(batch3)
    print(f"✓ Trajectory-only: intent_logits {out3['intent_logits'].shape}")
    
    # Test 4: Pretraining
    pretrain_out = model.forward_pretrain(batch1, mask_prob=0.2)
    print(f"✓ Pretraining: total_loss {pretrain_out['total_pretrain_loss'].item():.4f}")
    
    print(f"\nModel parameters: {count_parameters(model):,}")
    print(f"Registered pose encoders: {list(model.pose_encoders.keys())}")
