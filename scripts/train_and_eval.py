import argparse
import random
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformer_model import TransformerModel, count_parameters
from data_utils import MultimodalWindowDataset, collate_fn
import numpy as np


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(y_true, y_pred):
    """Compute binary classification metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / max((tp + tn + fp + fn), 1)
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc}


def focal_loss_logits(logits, targets, gamma=2.0, alpha=0.25, eps=1e-6):
    """Focal loss for binary classification from logits.
    
    Args:
        logits: (N,) raw model outputs
        targets: (N,) float in {0,1}
        gamma: focusing parameter (default 2.0)
        alpha: class weight for positive class (default 0.25)
        eps: numerical stability epsilon
    """
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = -alpha_t * ((1 - p_t) ** gamma) * torch.log(p_t + eps)
    return loss.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='datasets/npz')
    p.add_argument('--seq_len', type=int, default=30)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default='auto')
    p.add_argument('--train-steps', type=int, default=300, help='max training batches per epoch (increased from 200 to see more data)')
    p.add_argument('--epochs', type=int, default=10, help='number of epochs')
    p.add_argument('--lr', type=float, default=3e-4, help='learning rate (reduced from 1e-3 for stability)')
    p.add_argument('--weight-decay', type=float, default=1e-4, help='AdamW weight decay')
    p.add_argument('--grad-clip', type=float, default=0.5, help='gradient clipping norm (reduced from 1.0 for stability)')
    p.add_argument('--weighted-sampler', action='store_true', help='use WeightedRandomSampler to balance classes by future_label')
    p.add_argument('--class-weighted', action='store_true', help='use class weights for CrossEntropy on the main intent head')
    p.add_argument('--use-focal', action='store_true', help='use focal loss for future interaction auxiliary head')
    p.add_argument('--focal-gamma', type=float, default=2.0, help='focal loss gamma parameter')
    p.add_argument('--focal-alpha', type=float, default=0.25, help='focal loss alpha parameter')
    p.add_argument('--aux-weight', type=float, default=0.5, help='weight for auxiliary future prediction loss (reduced from 1.0)')
    p.add_argument('--checkpoint-dir', default='checkpoints', help='where to save model checkpoints')
    p.add_argument('--save-best-only', action='store_true', help='only save checkpoint for best validation F1')
    p.add_argument('--early-stop-patience', type=int, default=10, help='early stopping patience (increased from 0)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--d-model', type=int, default=128, help='model dimension (reduced from 256 for stability)')
    p.add_argument('--nhead', type=int, default=4, help='number of attention heads (reduced from 8)')
    p.add_argument('--num-layers', type=int, default=3, help='number of transformer layers (reduced from 4)')
    p.add_argument('--pretrained-ckpt', type=str, default=None, help='path to pretrained model checkpoint to load encoder weights')
    args = p.parse_args()

    set_seed(args.seed)
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print('Using device', device)

    root = Path(args.data)
    files = sorted([str(p) for p in root.rglob('*.npz')])
    if len(files) == 0:
        raise RuntimeError('No npz files found under ' + str(root))

    random.shuffle(files)
    split = int(0.9 * len(files))
    train_files = files[:split]
    val_files = files[split:]
    print(f'Found {len(files)} files; train {len(train_files)} val {len(val_files)}')

    train_ds = MultimodalWindowDataset(args.data, seq_len=args.seq_len, stride=args.seq_len // 2, files=train_files, backend='npz')
    val_ds = MultimodalWindowDataset(args.data, seq_len=args.seq_len, stride=args.seq_len // 2, files=val_files, backend='npz')

    # optionally build a weighted sampler to balance by future_label (our main prediction target)
    from torch.utils.data import WeightedRandomSampler
    if args.weighted_sampler:
        print('Building label weights for WeightedRandomSampler (this may take a moment)')
        labels = []
        for i in range(len(train_ds)):
            lbl = int(train_ds[i]['future_label'].item())  # Use future_label since that's what we're predicting
            labels.append(lbl)
        labels = np.array(labels)
        class_counts = np.bincount(labels, minlength=2)
        class_counts[class_counts == 0] = 1
        weights = np.array([1.0 / class_counts[l] for l in labels], dtype=np.float32)
        sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, collate_fn=collate_fn, num_workers=0, pin_memory=(device.type == 'cuda'))
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=(device.type == 'cuda'))

    # infer feature dims from a sample
    sample = train_ds[0]
    pose_feats = sample['pose'].shape[1] if sample['pose'].shape[1] > 0 else None
    traj_feats = sample['traj'].shape[1]
    
    # Check if dataset has only one modality (trajectory-only or pose-only)
    has_pose = pose_feats is not None and pose_feats > 0
    has_traj = traj_feats > 0
    is_single_modality = not (has_pose and has_traj)
    
    if is_single_modality:
        modality_name = "pose-only" if has_pose else "trajectory-only"
        print(f"\n⚠️  Dataset is {modality_name}!")
        print(f"   Self-supervised pretraining is recommended for single-modality datasets.")
        print(f"   Consider running: python pretrain.py --data {args.data} --epochs 20 --batch {args.batch}")
        print(f"   Then use: --pretrained-ckpt checkpoints/pretrain/best_model.pt\n")

    # compute class weights for CrossEntropy if requested
    class_weights = None
    if args.class_weighted:
        # build label counts over training windows for FUTURE_LABEL (main prediction target)
        lab_list = []
        for i in range(len(train_ds)):
            lab_list.append(int(train_ds[i]['future_label'].item()))
        lab_arr = np.array(lab_list)
        counts = np.bincount(lab_arr, minlength=2).astype(float)
        counts[counts == 0] = 1.0
        inv = 1.0 / counts
        # normalize to sum=2 (so weights are comparable)
        inv = inv * (2.0 / inv.sum())
        class_weights = torch.tensor(inv, dtype=torch.float32, device=device)
        print('Using class weights for CE:', class_weights.tolist())

    # Model with dynamic pose encoder support
    model = TransformerModel(
        pose_feats=pose_feats,  # Can be None for fully dynamic
        traj_feats=traj_feats, 
        d_model=args.d_model, 
        nhead=args.nhead, 
        num_layers=args.num_layers, 
        num_classes=2
    )
    model.to(device)
    print('Model params:', count_parameters(model))
    
    # Load pretrained weights if provided
    if args.pretrained_ckpt:
        print(f"\nLoading pretrained encoder from: {args.pretrained_ckpt}")
        try:
            ckpt = torch.load(args.pretrained_ckpt, map_location=device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                ckpt_state = ckpt['model_state_dict']
            else:
                ckpt_state = ckpt
            # Load only the encoder weights (pose_encoders, traj_encoder, transformer)
            # but not the classification heads
            model_state = model.state_dict()
            pretrained_state = {
                k: v for k, v in ckpt_state.items()
                if k in model_state and not k.startswith('intent_head') and not k.startswith('future_head')
            }
            model_state.update(pretrained_state)
            model.load_state_dict(model_state)
            print(f"✓ Loaded {len(pretrained_state)} pretrained parameters")
        except Exception as e:
            print(f"⚠️  Failed to load pretrained checkpoint: {e}")
            print("   Continuing with random initialization...")

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler - use exponential decay for stability
    # CosineAnnealing was causing instability (large LR jumps)
    from torch.optim.lr_scheduler import ExponentialLR
    scheduler = ExponentialLR(optim, gamma=0.95)
    
    # Checkpoint tracking
    best_val_f1 = -1.0
    patience_counter = 0
    
    # Ensure checkpoint directory exists
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # === Train ===
        model.train()
        train_steps = 0
        total_loss = 0.0
        total_loss_main = 0.0
        total_loss_aux = 0.0
        train_preds = []
        train_gts = []
        
        for batch in train_loader:
            if train_steps >= args.train_steps:
                break
                
            # Move to device
            for k in ['pose', 'traj', 'has_pose', 'has_traj', 'label', 'future_label']:
                if k in batch and torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)
            
            optim.zero_grad()
            # Reduce modality dropout or remove it - it hurts convergence
            out = model(batch, dropout_modality_probs=[0.0, 0.0])  # No dropout for better learning
            logits = out['intent_logits']
            # MAIN TASK: Predict FUTURE interaction (will they interact in next H seconds?)
            # This is the useful prediction for robot decision-making
            labels = batch['future_label']
            
            # Main intent head loss
            # Use label smoothing to prevent overconfident predictions
            if class_weights is not None:
                loss_main = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights, label_smoothing=0.1)
            else:
                loss_main = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.1)
            
            # Auxiliary current interaction head loss (helps learning but not main task)
            loss_aux = torch.tensor(0.0, device=device)
            if 'future_logits' in out and 'label' in batch:
                # Use future_logits to predict CURRENT interaction as auxiliary task
                fut_logits = out['future_logits'].squeeze(-1)
                curr_lbl = batch['label'].float()
                
                if args.use_focal:
                    loss_aux = focal_loss_logits(fut_logits, curr_lbl, gamma=args.focal_gamma, alpha=args.focal_alpha)
                else:
                    # Compute pos_weight for imbalanced labels
                    pos = (curr_lbl == 1).float().sum()
                    neg = (curr_lbl == 0).float().sum()
                    pos_weight = (neg / (pos + 1e-6)) if pos > 0 else 1.0
                    # Fix: create tensor directly instead of from scalar
                    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, device=device))
                    loss_aux = bce(fut_logits, curr_lbl)
            
            loss = loss_main + args.aux_weight * loss_aux
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optim.step()

            total_loss += loss.item()
            total_loss_main += loss_main.item()
            total_loss_aux += loss_aux.item()
            preds = logits.argmax(dim=1).cpu().tolist()
            train_preds.extend(preds)
            train_gts.extend(labels.cpu().tolist())
            train_steps += 1
        
        avg_loss = total_loss / max(train_steps, 1)
        avg_loss_main = total_loss_main / max(train_steps, 1)
        avg_loss_aux = total_loss_aux / max(train_steps, 1)
        train_met = compute_metrics(train_gts, train_preds)
        current_lr = optim.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{args.epochs} train_steps={train_steps} lr={current_lr:.6f} '
              f'loss={avg_loss:.4f} (main={avg_loss_main:.4f}, aux={avg_loss_aux:.4f}) '
              f'acc={train_met["accuracy"]:.3f} f1={train_met["f1"]:.3f}')

        # === Validation ===
        model.eval()
        val_preds = []
        val_gts = []
        val_loss = 0.0
        val_loss_main = 0.0
        val_loss_aux = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                for k in ['pose', 'traj', 'has_pose', 'has_traj', 'label', 'future_label']:
                    if k in batch and torch.is_tensor(batch[k]):
                        batch[k] = batch[k].to(device)
                
                out = model(batch, dropout_modality_probs=[0.0, 0.0])
                logits = out['intent_logits']
                # MAIN TASK: Predict FUTURE interaction
                labels = batch['future_label']
                
                # Main loss
                if class_weights is not None:
                    loss_main = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights, label_smoothing=0.1)
                else:
                    loss_main = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.1)
                
                # Auxiliary loss (current interaction)
                loss_aux = torch.tensor(0.0, device=device)
                if 'future_logits' in out and 'label' in batch:
                    fut_logits = out['future_logits'].squeeze(-1)
                    curr_lbl = batch['label'].float()
                    
                    if args.use_focal:
                        loss_aux = focal_loss_logits(fut_logits, curr_lbl, gamma=args.focal_gamma, alpha=args.focal_alpha)
                    else:
                        pos = (curr_lbl == 1).float().sum()
                        neg = (curr_lbl == 0).float().sum()
                        pos_weight = (neg / (pos + 1e-6)) if pos > 0 else 1.0
                        bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, device=device))
                        loss_aux = bce(fut_logits, curr_lbl)
                
                val_loss += (loss_main + args.aux_weight * loss_aux).item()
                val_loss_main += loss_main.item()
                val_loss_aux += loss_aux.item()
                preds = logits.argmax(dim=1).cpu().tolist()
                val_preds.extend(preds)
                val_gts.extend(labels.cpu().tolist())
                val_steps += 1
        
        avg_val_loss = val_loss / max(val_steps, 1)
        avg_val_loss_main = val_loss_main / max(val_steps, 1)
        avg_val_loss_aux = val_loss_aux / max(val_steps, 1)
        val_met = compute_metrics(val_gts, val_preds)
        print(f'  val: loss={avg_val_loss:.4f} (main={avg_val_loss_main:.4f}, aux={avg_val_loss_aux:.4f}) '
              f'acc={val_met["accuracy"]:.3f} p={val_met["precision"]:.3f} '
              f'r={val_met["recall"]:.3f} f1={val_met["f1"]:.3f}')

        # Learning rate scheduling - step every epoch
        scheduler.step()

        # Checkpointing
        current_val_f1 = val_met['f1']
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            patience_counter = 0
            ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            torch.save({
                'model_state': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'best_val_f1': best_val_f1,
                'args': vars(args)
            }, ckpt_path)
            print(f'  ✓ Saved best model (F1={best_val_f1:.3f}) to {ckpt_path}')
        else:
            patience_counter += 1
            if not args.save_best_only:
                ckpt_path = os.path.join(args.checkpoint_dir, f'model_epoch{epoch+1}.pt')
                torch.save({
                    'model_state': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'val_f1': current_val_f1,
                    'args': vars(args)
                }, ckpt_path)
                print(f'  Saved checkpoint to {ckpt_path}')

        # Early stopping
        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(f'Early stopping triggered after {patience_counter} epochs without improvement')
            break

    print(f'\nTraining complete! Best validation F1: {best_val_f1:.3f}')

    # Final evaluation on best model
    print('\nLoading best model for final evaluation...')
    print('MAIN TASK: Predicting FUTURE interaction (will person interact in next H seconds?)')
    best_ckpt = torch.load(os.path.join(args.checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(best_ckpt['model_state'])
    model.eval()
    
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in val_loader:
            for k in ['pose', 'traj', 'has_pose', 'has_traj', 'label', 'future_label']:
                if k in batch and torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)
            out = model(batch, dropout_modality_probs=[0.0, 0.0])
            logits = out['intent_logits']
            preds = logits.argmax(dim=1).cpu().tolist()
            lbls = batch['future_label'].cpu().tolist()  # Use future_label - that's what we're predicting!
            y_true.extend(lbls)
            y_pred.extend(preds)

    final_metrics = compute_metrics(y_true, y_pred)
    print('\nFinal validation metrics (best model):')
    for k, v in final_metrics.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')


if __name__ == '__main__':
    main()
