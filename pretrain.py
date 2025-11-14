"""Pretraining script using masked modality prediction and contrastive learning.

Uses the model's built-in pretraining mode for self-supervised learning.
This improves encoder quality before fine-tuning on labeled data.
"""

import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from data_utils import MultimodalWindowDataset, collate_fn
from transformer_model import TransformerModel, count_parameters


TENSOR_KEYS = [
    'pose',
    'gaze',
    'emotion',
    'traj',
    'robot',
    'modality_mask',
    'frame_labels',
    'label',
    'future_label',
    'intent_label',
    'has_pose',
    'has_gaze',
    'has_emotion',
    'has_traj',
    'has_robot',
]


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pretrain(args):
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    print(f'Pretraining on device: {device}')
    
    # Load dataset
    root = Path(args.data)
    files = sorted([str(p) for p in root.rglob('*.npz')])
    if len(files) == 0:
        raise RuntimeError(f'No .npz files found under {root}')
    
    print(f'Found {len(files)} files')
    random.shuffle(files)
    
    # For pretraining, use all data (no train/val split needed)
    dataset = MultimodalWindowDataset(args.data, seq_len=args.seq_len, stride=args.stride, files=files, backend='npz')
    loader = DataLoader(
        dataset, 
        batch_size=args.batch, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    
    # Infer dimensions from sample
    sample = dataset[0]
    pose_feats = sample['pose'].shape[1] if sample['pose'].shape[1] > 0 else None
    traj_feats = sample['traj'].shape[1]
    
    # Create model
    model = TransformerModel(
        pose_feats=pose_feats,
        traj_feats=traj_feats,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_classes=2
    )
    model.to(device)
    print(f'Model parameters: {count_parameters(model):,}')
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        total_recon_loss = 0.0
        total_contrastive_loss = 0.0
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(loader):
            # Move to device
            for key in TENSOR_KEYS:
                if key in batch and torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            
            optimizer.zero_grad()
            
            # Pretraining forward pass
            outputs = model.forward_pretrain(batch, mask_prob=args.mask_prob)
            
            loss = outputs['total_pretrain_loss']
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            total_recon_loss += outputs['recon_loss'].item()
            total_contrastive_loss += outputs['contrastive_loss'].item()
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % args.log_interval == 0:
                print(f'Epoch {epoch+1}/{args.epochs} [{batch_idx}/{len(loader)}] '
                      f'Loss: {loss.item():.4f} (recon: {outputs["recon_loss"].item():.4f}, '
                      f'contrast: {outputs["contrastive_loss"].item():.4f})')
            
            if batch_count >= args.max_batches:
                break
        
        avg_loss = total_loss / batch_count
        avg_recon = total_recon_loss / batch_count
        avg_contrast = total_contrastive_loss / batch_count
        
        print(f'\nEpoch {epoch+1}/{args.epochs} Summary:')
        print(f'  Total Loss: {avg_loss:.4f}')
        print(f'  Reconstruction Loss: {avg_recon:.4f}')
        print(f'  Contrastive Loss: {avg_contrast:.4f}')
        
        # Save checkpoint
        ckpt_dir: Path = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            ckpt_path = ckpt_dir / 'pretrain_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'args': vars(args)
            }, ckpt_path)
            print(f'  ✓ Saved best model (loss={best_loss:.4f}) to {ckpt_path}')
        else:
            patience_counter += 1
        
        if not args.save_best_only:
            ckpt_path = ckpt_dir / f'pretrain_epoch{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': vars(args)
            }, ckpt_path)
        
        # Early stopping
        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(f'\nEarly stopping after {patience_counter} epochs without improvement')
            break
    
    print(f'\nPretraining complete! Best loss: {best_loss:.4f}')
    best_model_path = ckpt_dir / "pretrain_best.pt"
    print(f'Pretrained model saved to: {best_model_path}')
    print('\nTo use pretrained weights, load them in training script:')
    print(f'  checkpoint = torch.load("{best_model_path}")')
    print('  model.load_state_dict(checkpoint["model_state_dict"])')


def main():
    parser = argparse.ArgumentParser(description='Pretrain multimodal transformer')
    parser.add_argument('--data', default='datasets/npz', help='Path to NPZ dataset')
    parser.add_argument('--seq-len', type=int, default=30, help='Sequence length')
    parser.add_argument('--stride', type=int, default=15, help='Window stride')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--max-batches', type=int, default=500, help='Max batches per epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--mask-prob', type=float, default=0.15, help='Modality masking probability')
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--checkpoint-dir', default='checkpoints/pretrain', help='Checkpoint directory')
    parser.add_argument('--save-best-only', action='store_true', help='Only save best checkpoint')
    parser.add_argument('--early-stop-patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--log-interval', type=int, default=20, help='Log every N batches')
    parser.add_argument('--device', default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    pretrain(args)


if __name__ == '__main__':
    main()
