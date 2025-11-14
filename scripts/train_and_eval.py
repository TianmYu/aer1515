import argparse
import random
import os
import sys
import math
from pathlib import Path
from typing import Dict

# Add parent directory to path so we can import from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformer_model import TransformerModel, count_parameters
from lstm_model import LSTMFusionModel
from data_utils import MultimodalWindowDataset, collate_fn
import numpy as np


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
    'sequence_label',
    'has_pose',
    'has_gaze',
    'has_emotion',
    'has_traj',
    'has_robot',
    'source_index',
    'window_index',
]


def get_intent_targets(batch):
    if 'intent_label' in batch:
        return batch['intent_label']
    if 'future_label' in batch:
        return batch['future_label']
    raise KeyError('Batch must contain intent_label or future_label')


def strip_to_pose_only(batch):
    drop_modalities = ['gaze', 'emotion', 'traj', 'robot']
    for key in drop_modalities:
        batch.pop(key, None)
    for key in ['has_gaze', 'has_emotion', 'has_traj', 'has_robot']:
        batch.pop(key, None)
    mask = batch.get('modality_mask')
    if torch.is_tensor(mask):
        # Keep pose mask and zero-out other modalities so encoders can ignore padding.
        pose_mask = mask[..., :1]
        zeros = torch.zeros_like(mask[..., 1:])
        batch['modality_mask'] = torch.cat([pose_mask, zeros], dim=-1)
    return batch


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


def format_confusion_counts(metrics: Dict[str, float]) -> str:
    return f"TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}"


def aggregate_by_sequence(preds, labels, sequence_indices, reduction: str):
    grouped_preds = {}
    grouped_labels = {}
    for pred, label, seq_idx in zip(preds, labels, sequence_indices):
        seq_idx = int(seq_idx)
        grouped_preds.setdefault(seq_idx, []).append(int(pred))
        grouped_labels.setdefault(seq_idx, int(label))

    agg_preds = []
    agg_labels = []
    for seq_idx, values in grouped_preds.items():
        if reduction == 'majority':
            agg_pred = 1 if sum(values) >= (len(values) / 2.0) else 0
        else:
            agg_pred = 1 if any(values) else 0
        agg_preds.append(agg_pred)
        agg_labels.append(grouped_labels.get(seq_idx, 0))
    return agg_labels, agg_preds


def move_batch_to_device(batch, device, pose_only: bool):
    if pose_only:
        strip_to_pose_only(batch)
    for key in TENSOR_KEYS:
        tensor = batch.get(key)
        if torch.is_tensor(tensor):
            batch[key] = tensor.to(device)
    return batch


def compute_losses(out, batch, class_weights, args, device):
    logits = out['intent_logits']
    labels = get_intent_targets(batch)
    smoothing = getattr(args, 'label_smoothing', 0.0)
    if class_weights is not None:
        loss_main = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights, label_smoothing=smoothing)
    else:
        loss_main = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=smoothing)

    loss_aux = torch.tensor(0.0, device=device)
    if 'future_logits' in out and ('intent_label' in batch or 'future_label' in batch or 'label' in batch):
        fut_logits = out['future_logits']
        future_targets = batch.get('intent_label', batch.get('future_label', batch.get('label'))).float()
        if fut_logits.ndim == 2 and fut_logits.shape[1] == 1:
            fut_logits = fut_logits.squeeze(-1)
        if future_targets.ndim == 1 and fut_logits.ndim == 2:
            future_targets = future_targets.unsqueeze(-1).expand_as(fut_logits)
        if future_targets.ndim == 2 and fut_logits.ndim == 1:
            future_targets = future_targets.squeeze(-1)
        if args.use_focal:
            loss_aux = focal_loss_logits(fut_logits, future_targets, gamma=args.focal_gamma, alpha=args.focal_alpha)
        else:
            pos = (future_targets == 1).float().sum()
            neg = (future_targets == 0).float().sum()
            pos_weight = (neg / (pos + 1e-6)) if pos > 0 else 1.0
            bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, device=device))
            loss_aux = bce(fut_logits, future_targets)
    return loss_main, loss_aux


def summarize_dataset(ds: MultimodalWindowDataset, name: str) -> None:
    infos = ds.file_infos
    if not infos:
        print(f'{name}: no usable sequences (length < seq_len)')
        return
    seq_total = len(infos)
    seq_pos = sum(info.get('intent_label', 0) for info in infos)
    frame_total = sum(info.get('frame_total', 0) for info in infos)
    frame_pos = sum(info.get('frame_positive', 0) for info in infos)
    windows = len(ds)
    seq_pct = (seq_pos / seq_total * 100.0) if seq_total else 0.0
    frame_pct = (frame_pos / frame_total * 100.0) if frame_total else 0.0
    print(f'{name}: sequences={seq_total} pos={seq_pos} ({seq_pct:.1f}%) '
          f'frames_pos={frame_pos}/{frame_total} ({frame_pct:.1f}%) windows={windows}')


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
    p.add_argument('--data', default='datasets/processed/dataverse_npz')
    p.add_argument('--seq_len', type=int, default=30)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default='auto')
    p.add_argument('--train-steps', type=int, default=None, help='max training batches per epoch (default: cover entire dataset)')
    p.add_argument('--epochs', type=int, default=10, help='number of epochs')
    p.add_argument('--val-ratio', type=float, default=0.1, help='fraction of files reserved for validation')
    p.add_argument('--stride', type=int, default=None, help='window stride (default: seq_len//2)')
    p.add_argument('--lr', type=float, default=3e-4, help='learning rate (reduced from 1e-3 for stability)')
    p.add_argument('--weight-decay', type=float, default=1e-4, help='AdamW weight decay')
    p.add_argument('--grad-clip', type=float, default=0.5, help='gradient clipping norm (reduced from 1.0 for stability)')
    p.add_argument('--weighted-sampler', action='store_true', help='use WeightedRandomSampler to balance classes by intent_label')
    p.add_argument('--class-weighted', action='store_true', help='use class weights for CrossEntropy on the main intent head')
    p.add_argument('--use-focal', action='store_true', help='use focal loss for future interaction auxiliary head')
    p.add_argument('--focal-gamma', type=float, default=2.0, help='focal loss gamma parameter')
    p.add_argument('--focal-alpha', type=float, default=0.25, help='focal loss alpha parameter')
    p.add_argument('--aux-weight', type=float, default=0.5, help='weight for auxiliary future prediction loss (reduced from 1.0)')
    p.add_argument('--checkpoint-dir', default='checkpoints', help='where to save model checkpoints')
    p.add_argument('--save-best-only', action='store_true', help='only save checkpoint for best validation F1')
    p.add_argument('--early-stop-patience', type=int, default=10, help='early stopping patience (increased from 0)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model-type', choices=['transformer', 'lstm'], default='transformer', help='backbone architecture to use')
    p.add_argument('--d-model', type=int, default=128, help='model dimension (reduced from 256 for stability)')
    p.add_argument('--nhead', type=int, default=4, help='number of attention heads (transformer only)')
    p.add_argument('--num-layers', type=int, default=3, help='number of transformer/LSTM layers')
    p.add_argument('--lstm-bidirectional', action='store_true', help='use bidirectional LSTM fusion (only for model-type=lstm)')
    p.add_argument('--lstm-dropout', type=float, default=0.1, help='dropout applied between stacked LSTM layers')
    p.add_argument('--pretrained-ckpt', type=str, default=None, help='path to pretrained model checkpoint to load encoder weights')
    p.add_argument('--pose-only', action='store_true', help='mask out all non-pose modalities during training and evaluation')
    p.add_argument('--eval-level', choices=['window', 'sequence', 'both'], default='sequence', help='metric used for model selection')
    p.add_argument('--sequence-agg', choices=['any', 'majority'], default='majority', help='how to aggregate window predictions at sequence level')
    p.add_argument('--label-smoothing', type=float, default=0.0, help='label smoothing applied to main intent loss')
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
        suggestions = []
        for candidate in [Path('datasets/processed/dataverse_npz'), Path('datasets/processed/mint_npz')]:
            if candidate.exists() and any(candidate.rglob('*.npz')):
                suggestions.append(str(candidate))
        hint = f"No npz files found under {root}"
        if suggestions:
            hint += ". Consider pointing --data to one of: " + ", ".join(suggestions)
        raise RuntimeError(hint)

    random.shuffle(files)
    split = int((1.0 - args.val_ratio) * len(files))
    split = min(max(split, 1), len(files) - 1) if len(files) > 1 else len(files)
    train_files = files[:split]
    val_files = files[split:]
    print(f'Found {len(files)} files; train {len(train_files)} val {len(val_files)}')

    stride = args.stride if args.stride is not None else max(1, args.seq_len // 2)
    train_ds = MultimodalWindowDataset(args.data, seq_len=args.seq_len, stride=stride, files=train_files, backend='npz')
    val_ds = MultimodalWindowDataset(args.data, seq_len=args.seq_len, stride=stride, files=val_files, backend='npz')

    summarize_dataset(train_ds, 'Train split')
    summarize_dataset(val_ds, 'Val split')

    # optionally build a weighted sampler to balance by intent_label (our main prediction target)
    from torch.utils.data import WeightedRandomSampler
    if args.weighted_sampler:
        print('Building label weights for WeightedRandomSampler (this may take a moment)')
        labels = np.array(train_ds.get_window_label_list(label_key='intent_label'))
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
    # build label counts over training windows for intent_label (main prediction target)
        counts = train_ds.get_label_counts(label_key='intent_label', num_classes=2).astype(float)
        counts[counts == 0] = 1.0
        inv = 1.0 / counts
        # normalize to sum=2 (so weights are comparable)
        inv = inv * (2.0 / inv.sum())
        class_weights = torch.tensor(inv, dtype=torch.float32, device=device)
        print('Using class weights for CE:', class_weights.tolist())

    effective_train_steps = args.train_steps if args.train_steps is not None else math.ceil(len(train_ds) / args.batch)
    print(f'Training batches per epoch: {effective_train_steps}')

    if args.model_type == 'transformer':
        model = TransformerModel(
            pose_feats=pose_feats,
            traj_feats=traj_feats,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            num_classes=2,
        )
    else:
        model = LSTMFusionModel(
            pose_feats=pose_feats,
            traj_feats=traj_feats,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_classes=2,
            bidirectional=args.lstm_bidirectional,
            lstm_dropout=args.lstm_dropout,
        )
    model.to(device)
    print('Model params:', count_parameters(model))
    
    # Load pretrained weights if provided (transformer-only for now)
    if args.pretrained_ckpt:
        if args.model_type != 'transformer':
            print("⚠️  Pretrained checkpoints are only supported for the transformer backbone right now; skipping load.")
        else:
            print(f"\nLoading pretrained encoder from: {args.pretrained_ckpt}")
            try:
                ckpt = torch.load(args.pretrained_ckpt, map_location=device)
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    ckpt_state = ckpt['model_state_dict']
                else:
                    ckpt_state = ckpt
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
            if train_steps >= effective_train_steps:
                break
            move_batch_to_device(batch, device, args.pose_only)

            optim.zero_grad()
            out = model(batch)
            loss_main, loss_aux = compute_losses(out, batch, class_weights, args, device)
            loss = loss_main + args.aux_weight * loss_aux
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optim.step()

            total_loss += loss.item()
            total_loss_main += loss_main.item()
            total_loss_aux += loss_aux.item()
            preds = out['intent_logits'].argmax(dim=1).cpu().tolist()
            train_preds.extend(preds)
            train_gts.extend(get_intent_targets(batch).cpu().tolist())
            train_steps += 1

        avg_loss = total_loss / max(train_steps, 1)
        avg_loss_main = total_loss_main / max(train_steps, 1)
        avg_loss_aux = total_loss_aux / max(train_steps, 1)
        train_met = compute_metrics(train_gts, train_preds)
        current_lr = optim.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{args.epochs} train_steps={train_steps} lr={current_lr:.6f} '
              f'loss={avg_loss:.4f} (main={avg_loss_main:.4f}, aux={avg_loss_aux:.4f}) '
          f'acc={train_met["accuracy"]:.3f} f1={train_met["f1"]:.3f} | {format_confusion_counts(train_met)}')

        # === Validation ===
        model.eval()
        val_preds = []
        val_gts = []
        val_loss = 0.0
        val_loss_main = 0.0
        val_loss_aux = 0.0
        val_steps = 0
        val_seq_labels = []
        val_seq_indices = []

        with torch.no_grad():
            for batch in val_loader:
                move_batch_to_device(batch, device, args.pose_only)
                out = model(batch)
                loss_main, loss_aux = compute_losses(out, batch, class_weights, args, device)
                val_loss += (loss_main + args.aux_weight * loss_aux).item()
                val_loss_main += loss_main.item()
                val_loss_aux += loss_aux.item()
                preds = out['intent_logits'].argmax(dim=1).cpu().tolist()
                val_preds.extend(preds)
                val_gts.extend(get_intent_targets(batch).cpu().tolist())
                val_seq_labels.extend(batch['sequence_label'].cpu().tolist())
                val_seq_indices.extend(batch['source_index'].cpu().tolist())
                val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)
        avg_val_loss_main = val_loss_main / max(val_steps, 1)
        avg_val_loss_aux = val_loss_aux / max(val_steps, 1)
        window_metrics = compute_metrics(val_gts, val_preds)
        seq_metrics = None
        if args.eval_level in {'sequence', 'both'}:
            seq_true, seq_pred = aggregate_by_sequence(val_preds, val_seq_labels, val_seq_indices, args.sequence_agg)
            seq_metrics = compute_metrics(seq_true, seq_pred)

        print(f'  val: loss={avg_val_loss:.4f} (main={avg_val_loss_main:.4f}, aux={avg_val_loss_aux:.4f}) '
              f'win_acc={window_metrics["accuracy"]:.3f} win_f1={window_metrics["f1"]:.3f} '
              f'| {format_confusion_counts(window_metrics)}', end='')
        if seq_metrics is not None:
            seq_counts = format_confusion_counts(seq_metrics)
            print(f' || seq_acc={seq_metrics["accuracy"]:.3f} seq_f1={seq_metrics["f1"]:.3f} '
                  f'| {seq_counts}')
        else:
            print()

        # Learning rate scheduling - step every epoch
        scheduler.step()

        # Checkpointing
        current_val_f1 = seq_metrics['f1'] if seq_metrics is not None else window_metrics['f1']
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
    y_seq_labels = []
    y_seq_indices = []
    with torch.no_grad():
        for batch in val_loader:
            move_batch_to_device(batch, device, args.pose_only)
            out = model(batch)
            preds = out['intent_logits'].argmax(dim=1).cpu().tolist()
            lbls = get_intent_targets(batch).cpu().tolist()
            y_true.extend(lbls)
            y_pred.extend(preds)
            y_seq_labels.extend(batch['sequence_label'].cpu().tolist())
            y_seq_indices.extend(batch['source_index'].cpu().tolist())

    final_window = compute_metrics(y_true, y_pred)
    final_seq = None
    if args.eval_level in {'sequence', 'both'}:
        seq_true, seq_pred = aggregate_by_sequence(y_pred, y_seq_labels, y_seq_indices, args.sequence_agg)
        final_seq = compute_metrics(seq_true, seq_pred)

    print('\nFinal validation metrics (best model):')
    print('  Window-level:')
    for k, v in final_window.items():
        print(f'    {k}: {v:.4f}' if isinstance(v, float) else f'    {k}: {v}')
    if final_seq is not None:
        print('  Sequence-level:')
        for k, v in final_seq.items():
            print(f'    {k}: {v:.4f}' if isinstance(v, float) else f'    {k}: {v}')


if __name__ == '__main__':
    main()
