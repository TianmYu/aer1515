"""Training script for intent prediction from 3D pose sequences."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from transformer_model_simple import SimpleTransformerIntentClassifier, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata(file_paths: Sequence[str]) -> Dict[str, Dict[str, int]]:
    metadata: Dict[str, Dict[str, int]] = {}
    for path in file_paths:
        with np.load(path, allow_pickle=True) as data:
            metadata[path] = {
                'label': int(data['intent_label']),
                'length': int(data['pose'].shape[0]),
            }
    return metadata


def stratified_split(
    files: Sequence[str],
    metadata: Dict[str, Dict[str, int]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    positives = [f for f in files if metadata[f]['label'] == 1]
    negatives = [f for f in files if metadata[f]['label'] == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def split_group(group: List[str]) -> Tuple[List[str], List[str]]:
        if not group:
            return [], []
        val_count = int(round(len(group) * val_ratio))
        if len(group) > 0 and val_ratio > 0 and val_count == 0:
            val_count = 1
        val = group[:val_count]
        train = group[val_count:]
        return train, val

    train_pos, val_pos = split_group(positives)
    train_neg, val_neg = split_group(negatives)

    train = train_pos + train_neg
    val = val_pos + val_neg
    rng.shuffle(train)
    rng.shuffle(val)

    if not train or not val:
        split = int((1.0 - val_ratio) * len(files))
        train = list(files[:split])
        val = list(files[split:])

    return train, val


class FullSequenceDataset(Dataset):
    """Dataset that loads full sequences without sliding windows."""

    def __init__(self, npz_files: Sequence[str], max_seq_len: int = 0) -> None:
        self.files = list(npz_files)
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with np.load(self.files[idx], allow_pickle=True) as data:
            pose = torch.from_numpy(data['pose']).float()
            label = int(data['intent_label'])

        if self.max_seq_len > 0 and pose.shape[0] > self.max_seq_len:
            pose = pose[: self.max_seq_len]

        return {
            'pose': pose,
            'intent_label': torch.tensor(label, dtype=torch.long),
            'length': torch.tensor(pose.shape[0], dtype=torch.long),
        }


def collate_full_sequences(batch: Sequence[Dict[str, torch.Tensor]], pad_value: float = 0.0):
    if not batch:
        raise ValueError('Batch is empty')

    max_len = max(item['pose'].shape[0] for item in batch)
    batch_size = len(batch)

    pose_padded = torch.full((batch_size, max_len, 68), pad_value, dtype=torch.float32)
    labels = torch.zeros(batch_size, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item['pose'].shape[0]
        pose_padded[i, :seq_len] = item['pose']
        labels[i] = item['intent_label']
        lengths[i] = seq_len

    return {'pose': pose_padded, 'intent_label': labels, 'lengths': lengths}


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
    tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())
    fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / max((tp + tn + fp + fn), 1)
    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'accuracy': acc,
    }


def summarize_split(name: str, files: Sequence[str], metadata: Dict[str, Dict[str, int]]):
    if not files:
        print(f'{name}: 0 sequences (empty split)')
        return np.array([]), np.array([])

    labels = np.array([metadata[f]['label'] for f in files])
    lengths = np.array([metadata[f]['length'] for f in files])
    print(
        f'{name}: {len(files)} sequences | +{labels.sum()} ({100 * labels.mean():.1f}% pos) '
        f'| mean len {lengths.mean():.1f} ± {lengths.std():.1f} | median {np.median(lengths):.1f}'
    )
    return labels, lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='datasets/processed_filtered/dataverse_npz')
    parser.add_argument('--batch', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--checkpoint_dir', default='checkpoints/simple')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--nhead', type=int, default=6)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--downsample_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--disable_velocity', action='store_true')
    parser.add_argument('--max_seq_len', type=int, default=0, help='Optional cap on timesteps (0=full sequence)')
    parser.add_argument('--early_stop_patience', type=int, default=8)
    parser.add_argument('--min_epochs', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--class_weighting', action='store_true', help='Use inverse frequency class weights')
    parser.add_argument('--scheduler', choices=['none', 'cosine'], default='cosine')
    parser.add_argument('--use_amp', action='store_true', help='Enable mixed precision training when CUDA is available')
    parser.add_argument('--log_history', default='metrics_history.json')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = args.use_amp and device.type == 'cuda'
    print(f'Using device: {device} (AMP={use_amp})')

    root = Path(args.data)
    files = sorted(str(p) for p in root.glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'No NPZ files found under {root}')

    metadata = load_metadata(files)
    train_files, val_files = stratified_split(files, metadata, args.val_ratio, args.seed)
    print(f'Train files: {len(train_files)}, Val files: {len(val_files)}')

    train_labels, _ = summarize_split('Train', train_files, metadata)
    val_labels, _ = summarize_split('Val', val_files, metadata)

    train_ds = FullSequenceDataset(train_files, max_seq_len=args.max_seq_len)
    val_ds = FullSequenceDataset(val_files, max_seq_len=args.max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate_full_sequences,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=collate_full_sequences,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    model = SimpleTransformerIntentClassifier(
        pose_dim=68,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        downsample_layers=args.downsample_layers,
        use_velocity_features=not args.disable_velocity,
        dropout=args.dropout,
    )
    model.to(device)
    print(f'Model parameters: {count_parameters(model):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    class_weights_tensor = None
    if args.class_weighting and len(train_labels) > 0:
        class_counts = np.bincount(train_labels.astype(int), minlength=2)
        weights = class_counts.sum() / (2.0 * np.clip(class_counts, 1, None))
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f'Class weights: {weights}')

    scaler = GradScaler(enabled=use_amp)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = ckpt_dir / args.log_history

    best_val_f1 = -1.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds: List[int] = []
        train_gts: List[int] = []

        for batch in train_loader:
            pose = batch['pose'].to(device)
            labels = batch['intent_label'].to(device)
            lengths = batch['lengths'].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(pose, lengths=lengths)
                loss = F.cross_entropy(logits, labels, weight=class_weights_tensor)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
            train_gts.extend(labels.detach().cpu().tolist())

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = train_loss / max(1, len(train_loader))
        train_metrics = compute_metrics(train_gts, train_preds)

        model.eval()
        val_loss = 0.0
        val_preds: List[int] = []
        val_gts: List[int] = []

        with torch.no_grad():
            for batch in val_loader:
                pose = batch['pose'].to(device)
                labels = batch['intent_label'].to(device)
                lengths = batch['lengths'].to(device)

                with autocast(enabled=use_amp):
                    logits = model(pose, lengths=lengths)
                    loss = F.cross_entropy(logits, labels, weight=class_weights_tensor)

                val_loss += loss.item()
                val_preds.extend(logits.argmax(dim=1).cpu().tolist())
                val_gts.extend(labels.cpu().tolist())

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_metrics = compute_metrics(val_gts, val_preds)

        lr = optimizer.param_groups[0]['lr']
        print(
            f'\nEpoch {epoch}/{args.epochs} | lr={lr:.2e}\n'
            f'  Train: loss={avg_train_loss:.4f} acc={train_metrics["accuracy"]:.3f} '
            f'f1={train_metrics["f1"]:.3f} (TP={train_metrics["tp"]} FP={train_metrics["fp"]} '
            f'TN={train_metrics["tn"]} FN={train_metrics["fn"]})\n'
            f'  Val:   loss={avg_val_loss:.4f} acc={val_metrics["accuracy"]:.3f} '
            f'f1={val_metrics["f1"]:.3f} (TP={val_metrics["tp"]} FP={val_metrics["fp"]} '
            f'TN={val_metrics["tn"]} FN={val_metrics["fn"]})'
        )

        history.append(
            {
                'epoch': epoch,
                'lr': lr,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }
        )

        with history_path.open('w') as f:
            json.dump(history, f, indent=2)

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            ckpt_path = ckpt_dir / 'best_model.pt'
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': best_val_f1,
                    'args': vars(args),
                },
                ckpt_path,
            )
            print(f'  ✓ Saved best model (F1={best_val_f1:.3f})')
        else:
            patience_counter += 1
            if epoch >= args.min_epochs and patience_counter >= args.early_stop_patience:
                print(
                    f'\n⚠ Early stopping: no improvement for {args.early_stop_patience} epochs '
                    f'after epoch {epoch}'
                )
                break

    print(f'\nTraining complete! Best val F1: {best_val_f1:.3f}')
    print(f'Best checkpoint: {ckpt_dir / "best_model.pt"}')


if __name__ == '__main__':
    main()
