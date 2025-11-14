"""Multimodal transformer for intent-to-interact prediction with modality masking.

The model encodes pose, gaze, emotion, trajectory, robot state, and future modalities
through lightweight temporal encoders, fuses them with a transformer encoder, and
supports self-supervised pretraining via masked modality reconstruction. Modalities
can be missing at training or inference time; zeroed tokens plus attention masking
handle absent signals gracefully so heterogeneous datasets can be combined.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

GAZE_DIM = 5
EMOTION_DIM = 15
TRAJ_DIM = 6
ROBOT_DIM = 6


class TemporalEncoder(nn.Module):
    """Temporal encoder: conv -> conv -> masked global average pooling."""

    def __init__(self, in_feats: int, d_model: int = 128):
        super().__init__()
        self.out_dim = d_model
        self.conv1 = nn.Conv1d(in_feats, d_model // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TemporalEncoder expects (B,T,F) but received shape {tuple(x.shape)}")
        if x.shape[1] == 0:
            return torch.zeros(x.shape[0], self.out_dim, device=x.device, dtype=x.dtype)

        mask_tensor: Optional[torch.Tensor] = None
        if mask is not None:
            if mask.ndim == 3:
                mask_tensor = mask.squeeze(-1)
            elif mask.ndim == 2:
                mask_tensor = mask
            else:
                raise ValueError(f"Mask must be (B,T) or (B,T,1); got {tuple(mask.shape)}")
            if mask_tensor.shape[:2] != x.shape[:2]:
                raise ValueError("Mask and input sequence length must match for TemporalEncoder")
            mask_tensor = mask_tensor.to(x.device, dtype=x.dtype).clamp(0.0, 1.0)
            x = x * mask_tensor.unsqueeze(-1)

        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        if mask_tensor is not None:
            masked = x * mask_tensor.unsqueeze(1)
            lengths = mask_tensor.sum(dim=1, keepdim=True).clamp_min(1.0)
            x = masked.sum(dim=2) / lengths
        else:
            x = self.pool(x).squeeze(-1)
        return x


@dataclass
class ModalitySpec:
    """Configuration for a modality stream."""

    name: str
    input_dim: Optional[int] = None
    dynamic: bool = False
    dropout_prob: float = 0.0
    required: bool = False


class ModalityEncoder(nn.Module):
    """Wraps temporal encoders for fixed or dynamic-width modalities."""

    def __init__(self, spec: ModalitySpec, d_model: int):
        super().__init__()
        self.spec = spec
        self.d_model = d_model
        self.dynamic = spec.dynamic
        if self.dynamic:
            self.encoders = nn.ModuleDict()
            self.projections = nn.ModuleDict()
        else:
            in_feats = max(1, spec.input_dim or 1)
            self.encoder = TemporalEncoder(in_feats, d_model)
            self.projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected (B,T,F) or (B,F) for modality '{self.spec.name}', got {tuple(x.shape)}")
        if x.shape[1] == 0 or x.shape[2] == 0:
            return torch.zeros(x.shape[0], self.d_model, device=x.device, dtype=x.dtype)
        if self.dynamic:
            dim = x.shape[2]
            key = str(dim)
            if key not in self.encoders:
                self.encoders[key] = TemporalEncoder(max(1, dim), self.d_model)
                self.projections[key] = nn.Linear(self.d_model, self.d_model)
            encoder = self.encoders[key]
            projection = self.projections[key]
            if next(encoder.parameters()).device != x.device:
                self.encoders[key] = encoder.to(x.device)
                self.projections[key] = projection.to(x.device)
            encoder = self.encoders[key]
            projection = self.projections[key]
        else:
            encoder = self.encoder
            projection = self.projection
            if next(encoder.parameters()).device != x.device:
                self.encoder = encoder.to(x.device)
                self.projection = projection.to(x.device)
                encoder = self.encoder
                projection = self.projection
        out = encoder(x.float(), mask=mask)
        out = projection(out)
        return out


class TransformerModel(nn.Module):
    """Multimodal fusion transformer with modality-aware masking."""

    def __init__(
        self,
        pose_feats: Optional[int] = None,
    traj_feats: int = TRAJ_DIM,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        num_classes: int = 2,
        future_horizons: Sequence[float] = (2.0,),
        modality_specs: Optional[Sequence[ModalitySpec]] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.future_horizons = list(future_horizons)

        if modality_specs is None:
            modality_specs = self._build_default_specs(pose_feats, traj_feats)
        if not modality_specs:
            raise ValueError("At least one modality specification is required")

        self.modality_specs: Dict[str, ModalitySpec] = {spec.name: spec for spec in modality_specs}
        self.modality_order: List[str] = [spec.name for spec in modality_specs]
        self.modality_encoders = nn.ModuleDict({spec.name: ModalityEncoder(spec, d_model) for spec in modality_specs})
        self.modality_tokens = nn.ParameterDict({spec.name: nn.Parameter(torch.randn(d_model)) for spec in modality_specs})

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="relu",
            batch_first=False,
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.intent_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))
        self.future_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, len(self.future_horizons)))

        self.mask_prediction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.contrastive_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
        )

    @staticmethod
    def _build_default_specs(pose_feats: Optional[int], traj_feats: int) -> List[ModalitySpec]:
        return [
            ModalitySpec(name="pose", input_dim=pose_feats, dynamic=True, dropout_prob=0.05, required=True),
            ModalitySpec(name="gaze", input_dim=GAZE_DIM, dropout_prob=0.1),
            ModalitySpec(name="emotion", input_dim=EMOTION_DIM, dropout_prob=0.05),
            ModalitySpec(name="traj", input_dim=max(1, traj_feats), dropout_prob=0.05),
            ModalitySpec(name="robot", input_dim=ROBOT_DIM, dropout_prob=0.05),
        ]

    def _normalize_dropout(
        self, dropout_modality_probs: Optional[Union[Dict[str, float], Sequence[float]]]
    ) -> Dict[str, float]:
        if dropout_modality_probs is None:
            return {}
        if isinstance(dropout_modality_probs, dict):
            return dict(dropout_modality_probs)
        mapping: Dict[str, float] = {}
        for idx, prob in enumerate(dropout_modality_probs):
            if idx >= len(self.modality_order):
                break
            mapping[self.modality_order[idx]] = prob
        return mapping

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        dropout_modality_probs: Optional[Union[Dict[str, float], Sequence[float]]] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        embeddings: List[Optional[torch.Tensor]] = [None] * len(self.modality_order)
        presence: List[Optional[torch.Tensor]] = [None] * len(self.modality_order)
        pending: List[int] = []
        batch_size: Optional[int] = None
        device: Optional[torch.device] = None

        mask_tensor = batch.get("modality_mask")
        modality_masks: Optional[Dict[str, torch.Tensor]] = None
        if mask_tensor is not None:
            if mask_tensor.ndim != 3 or mask_tensor.shape[2] < len(self.modality_order):
                raise ValueError(
                    "modality_mask must have shape (B, T, num_modalities) matching model configuration"
                )
            modality_masks = {}
            for idx, name in enumerate(self.modality_order):
                modality_masks[name] = mask_tensor[:, :, idx]

        for idx, name in enumerate(self.modality_order):
            tensor = batch.get(name)
            if tensor is None:
                pending.append(idx)
                continue
            x = tensor.float()
            if x.ndim == 2:
                x = x.unsqueeze(1)
            if x.ndim != 3:
                raise ValueError(f"Modality '{name}' must provide (B,T,F) or (B,F), got {tuple(x.shape)}")
            if batch_size is None:
                batch_size = x.shape[0]
                device = x.device
            elif x.shape[0] != batch_size:
                raise ValueError("All modality tensors must share the same batch size")
            encoder = self.modality_encoders[name]
            mask = modality_masks[name] if modality_masks is not None else None
            emb = encoder(x, mask=mask)
            emb = emb.to(device)
            has_key = f"has_{name}"
            if has_key in batch:
                modality_presence = batch[has_key].float().to(device)
                if modality_presence.ndim > 1:
                    modality_presence = modality_presence.view(batch_size, -1).mean(dim=1)
                modality_presence = modality_presence.clamp(0.0, 1.0)
            elif modality_masks is not None and mask is not None:
                modality_presence = mask.to(device).float().max(dim=1)[0].clamp(0.0, 1.0)
            else:
                modality_presence = torch.ones(batch_size, device=device)
            emb = emb * modality_presence.unsqueeze(-1)
            embeddings[idx] = emb
            presence[idx] = modality_presence

        if batch_size is None or device is None:
            raise ValueError("Batch must provide at least one modality tensor")

        for idx in pending:
            embeddings[idx] = torch.zeros(batch_size, self.d_model, device=device)
            presence[idx] = torch.zeros(batch_size, device=device)

        embeddings_tensors: List[torch.Tensor] = [
            emb if emb is not None else torch.zeros(batch_size, self.d_model, device=device)
            for emb in embeddings
        ]
        presence_tensors: List[torch.Tensor] = []
        for idx, pres in enumerate(presence):
            if pres is not None:
                presence_tensors.append(pres)
            else:
                if modality_masks is not None:
                    mask = modality_masks[self.modality_order[idx]].to(device).float()
                    presence_tensors.append(mask.max(dim=1)[0].clamp(0.0, 1.0))
                else:
                    presence_tensors.append(torch.zeros(batch_size, device=device))

        dropout = self._normalize_dropout(dropout_modality_probs)
        if self.training or dropout:
            for idx, name in enumerate(self.modality_order):
                spec = self.modality_specs[name]
                if getattr(spec, "required", False):
                    continue
                base_p = spec.dropout_prob if self.training else 0.0
                override_p = dropout.get(name, 0.0)
                p = 1.0 - (1.0 - base_p) * (1.0 - override_p)
                if p <= 0:
                    continue
                drop_mask = (torch.rand(batch_size, device=device) < p).float()
                embeddings_tensors[idx] = embeddings_tensors[idx] * (1.0 - drop_mask).unsqueeze(-1)
                presence_tensors[idx] = presence_tensors[idx] * (1.0 - drop_mask)

        cls = self.cls_token.to(device).expand(batch_size, -1, -1)
        tokens = [cls]
        for idx, name in enumerate(self.modality_order):
            token = embeddings_tensors[idx].unsqueeze(1) + self.modality_tokens[name].to(device).view(1, 1, -1)
            tokens.append(token)
        tokens = torch.cat(tokens, dim=1)

        src = tokens.permute(1, 0, 2)
        presence_stack = torch.stack(presence_tensors, dim=1).bool()
        mask = torch.zeros(batch_size, 1 + presence_stack.shape[1], dtype=torch.bool, device=device)
        mask[:, 1:] = ~presence_stack

        fused = self.fusion_transformer(src, src_key_padding_mask=mask)
        fused = fused.permute(1, 0, 2)
        cls_out = fused[:, 0, :]

        intent_logits = self.intent_head(cls_out)
        future_logits = self.future_head(cls_out)

        result: Dict[str, torch.Tensor] = {
            "intent_logits": intent_logits,
            "future_logits": future_logits,
        }

        if return_features:
            result["features"] = cls_out
            result["token_embeddings"] = fused
            result["modality_embeddings"] = torch.stack(embeddings_tensors, dim=1)
            result["presence"] = torch.stack(presence_tensors, dim=1)

        return result

    def forward_pretrain(
        self, batch: Dict[str, torch.Tensor], mask_prob: float = 0.15
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            ref = self.forward(batch, return_features=True)
            target_features = ref["features"].detach()

        masked = self.forward(
            batch,
            dropout_modality_probs={name: mask_prob for name in self.modality_order},
            return_features=True,
        )

        recon_pred = self.mask_prediction_head(masked["features"])
        recon_loss = F.mse_loss(recon_pred, target_features)

        proj_ref = self.contrastive_projection(target_features)
        proj_masked = self.contrastive_projection(masked["features"])
        contrastive_loss = F.mse_loss(proj_ref, proj_masked)

        return {
            "recon_loss": recon_loss,
            "contrastive_loss": contrastive_loss,
            "total_pretrain_loss": recon_loss + 0.5 * contrastive_loss,
            "features": masked["features"],
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(7)

    model = TransformerModel(pose_feats=51, traj_feats=TRAJ_DIM, d_model=64, nhead=4, num_layers=2, future_horizons=(2.0, 5.0))

    B, T = 4, 30
    batch_full = {
    "pose": torch.randn(B, T, 51),
    "gaze": torch.randn(B, T, GAZE_DIM),
    "emotion": torch.randn(B, T, EMOTION_DIM),
    "traj": torch.randn(B, T, TRAJ_DIM),
    "robot": torch.randn(B, T, ROBOT_DIM),
        "has_pose": torch.ones(B),
        "has_gaze": torch.ones(B),
        "has_emotion": torch.ones(B),
        "has_traj": torch.ones(B),
        "has_robot": torch.ones(B),
    }

    model.eval()
    out_full = model(batch_full)
    print(f"✓ full modalities: intent {out_full['intent_logits'].shape}, future {out_full['future_logits'].shape}")

    batch_pose4 = dict(batch_full)
    batch_pose4["pose"] = torch.randn(B, T, 4)
    batch_pose4["has_pose"] = torch.ones(B)
    out_pose4 = model(batch_pose4)
    print(f"✓ dynamic pose dim: intent {out_pose4['intent_logits'].shape}")

    batch_pose_only = {
    "pose": torch.randn(B, T, 51),
        "has_pose": torch.ones(B),
    }
    out_pose_only = model(batch_pose_only)
    print(f"✓ pose only batch: intent {out_pose_only['intent_logits'].shape}, future {out_pose_only['future_logits'].shape}")

    model.train()
    pretrain_out = model.forward_pretrain(batch_full, mask_prob=0.2)
    print(f"✓ pretraining losses: total {pretrain_out['total_pretrain_loss'].item():.4f}")

    print(f"Model parameters: {count_parameters(model):,}")
    pose_encoder = model.modality_encoders["pose"]
    registered_dims = list(getattr(pose_encoder, "encoders", {}).keys())
    print(f"Registered pose encoder dims: {registered_dims}")
