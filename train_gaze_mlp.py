"""Train a simple MLP to predict gaze vector from face landmarks.

Usage (example):
    python face_and_pose/train_gaze_mlp.py --data-file face_and_pose/day09/p10.txt --epochs 10

This script expects the Mediapipe face-landmarker to be available and will attempt to
extract landmarks for each image path referenced in the annotation file. It is intentionally
simple and synchronous for quick experimentation.
"""
import argparse
import math
from pathlib import Path
import sys
import numpy as np
import os
import hashlib

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
except Exception:
    print('PyTorch not available. Please install torch to train the MLP.')
    raise

import gaze_mlp
import face_landmark as fl
import one_image_debug
import data_pipeline
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def make_face_landmark_extractor(model_path: str):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    def extractor(rel_path: str):
        # Build an absolute image path relative to this script's parent folder so the
        # extractor works regardless of the current working directory.
        base_dir = Path(__file__).resolve().parent
        p = Path(rel_path)
        # if caller passed an absolute path, use it directly
        if p.is_absolute():
            img_path = str(p)
        else:
            img_path = str(base_dir / rel_path)
        # If the image file doesn't exist locally, signal that to the caller so
        # the dataset builder can skip it cleanly.
        if not Path(img_path).exists():
            raise FileNotFoundError(f'Image file not found: {img_path}')
        img = mp.Image.create_from_file(img_path)
        res = detector.detect(img)
        if not hasattr(res, 'face_landmarks') or len(res.face_landmarks) == 0:
            return None
        face_landmarks = res.face_landmarks[0]
        # Map mediapipe landmarks to (x,y,z) list. Use normalized coords
        pts = []
        for lm in face_landmarks:
            pts.append((float(lm.x), float(lm.y), float(lm.z)))
        return pts

    return extractor


# Use centralized annotation collection from data_pipeline


def train(args):
    # Collect annotation lines from one or more files/directories
    ann_lines = []
    base_dir = Path(__file__).resolve().parent
    sources = []
    if getattr(args, 'data_files', None):
        sources.extend(args.data_files)
    elif getattr(args, 'data_file', None):
        sources.append(args.data_file)
    for s in sources:
        try:
            p = Path(s)
            ann_lines.extend(data_pipeline.collect_annotation_lines(p))
        except Exception:
            # if collecting fails for a specific source, continue with others
            print('Warning: failed to collect annotations from', s)
    
    # data_pipeline.collect_annotation_lines already resolves paths to absolute,
    # so no further path manipulation needed here
    extractor = make_face_landmark_extractor(args.model_asset_path)
    # Optional landmark caching to speed repeated dataset builds
    if getattr(args, 'cache_landmarks', False):
        cache_dir = Path('tmp_out') / 'landmarks'
        cache_dir.mkdir(parents=True, exist_ok=True)

        # keep reference to original extractor to avoid recursion
        base_extractor = extractor

        def cached_extractor(rel_path: str):
            # compute absolute image path same as extractor does
            base_dir = Path(__file__).resolve().parent
            p = Path(rel_path)
            img_path = str(p) if p.is_absolute() else str(base_dir / rel_path)
            if not Path(img_path).exists():
                raise FileNotFoundError(f'Image file not found: {img_path}')
            # build cache key using absolute path + mtime to allow invalidation
            try:
                mtime = os.path.getmtime(img_path)
            except Exception:
                mtime = 0
            key_src = f"{img_path}|{mtime}".encode('utf-8')
            key = hashlib.sha1(key_src).hexdigest()
            cache_file = cache_dir / f"{key}.npy"
            if cache_file.exists():
                try:
                    arr = np.load(str(cache_file))
                    return [tuple(map(float, row)) for row in arr.reshape(-1, 3)]
                except Exception:
                    # fallthrough to re-extract
                    pass
            # run actual extractor
            landmarks = base_extractor(rel_path)
            if landmarks is None:
                return None
            try:
                a = np.asarray(landmarks, dtype=np.float32)
                np.save(str(cache_file), a)
            except Exception:
                pass
            return landmarks

        extractor = cached_extractor
    # parse augmentations (comma-separated degrees) and pass to dataset builder
    rotations = None
    if getattr(args, 'augment_rotations', None):
        try:
            rotations = [int(x) for x in args.augment_rotations.split(',') if x.strip()]
            print('Applying rotations augmentation:', rotations)
        except Exception:
            rotations = None
    augment_scales_list = None
    if getattr(args, 'augment_scales', None):
        try:
            augment_scales_list = [float(x) for x in args.augment_scales.split(',') if x.strip()]
            print('Applying scale augmentation:', augment_scales_list, 'pivot=', args.augment_scale_pivot)
        except Exception:
            augment_scales_list = None
    # stash parsed scales into args for downstream convenience
    setattr(args, 'augment_scales_list', augment_scales_list)

    print('Building dataset (may take a while)...')
    # Track dataset source for each annotation line (for balancing)
    ann_dataset_labels = []
    for line in ann_lines:
        # Label as 'columbia' if path contains 'columbia', else 'mpii'
        ann_dataset_labels.append('columbia' if 'columbia' in line.lower() else 'mpii')
    
    result = gaze_mlp.build_dataset_from_annotations(
        ann_lines,
        extractor,
        rotations=rotations,
        scales=getattr(args, 'augment_scales_list', None),
        scale_pivot=getattr(args, 'augment_scale_pivot', 'center'),
        dataset_labels=ann_dataset_labels,
    )
    
    if len(result) == 3:
        X, Y, dataset_labels = result
    else:
        X, Y = result
        dataset_labels = ann_dataset_labels[:X.shape[0]]  # fallback
    
    if X.shape[0] == 0:
        print('No training data found (no faces detected). Exiting.')
        return
    
    print(f'Built dataset: {X.shape[0]} samples from {len(ann_lines)} annotations')
    
    # Print composition for debugging
    mpii_total = sum(1 for lbl in dataset_labels if lbl == 'mpii')
    columbia_total = sum(1 for lbl in dataset_labels if lbl == 'columbia')
    print(f'Dataset composition: {mpii_total} MPIIGaze, {columbia_total} Columbia')

    # If user requested a dry-run, stop after dataset building so they can
    # verify counts / samples without starting training (this avoids downstream
    # model/shape assumptions and expensive training work).
    if getattr(args, 'dry_run', False):
        print(f'Dry-run: collected {X.shape[0]} samples; feature dim={X.shape[1]}')
        return

    # Optional CSV logging setup
    log_csv_path = getattr(args, 'log_csv', None)
    if log_csv_path:
        try:
            from csv import writer as _csv_writer
            # Write header (overwrite existing file)
            with open(log_csv_path, 'w', newline='') as f:
                w = _csv_writer(f)
                w.writerow(['epoch', 'train_loss', 'val_loss', 'val_mean_angle_deg', 'mpii_mean_angle_deg', 'columbia_mean_angle_deg', 'lr'])
            print('Logging per-epoch metrics to', log_csv_path)
        except Exception as e:
            print('Failed to initialize CSV logging:', e)
            log_csv_path = None

    # simple train/val split using --val-split
    n = X.shape[0]
    idx = np.arange(n)
    np.random.seed(0)
    np.random.shuffle(idx)
    split = int(n * (1.0 - args.val_split))
    # ensure at least one sample in each split when possible
    if split <= 0 and n > 1:
        split = 1
    if split >= n and n > 1:
        split = n - 1
    train_idx, val_idx = idx[:split], idx[split:]

    in_dim = X.shape[1]

    # feature normalization (compute on train split)
    feat_mean = X[train_idx].mean(axis=0)
    feat_std = X[train_idx].std(axis=0)
    feat_std[feat_std < 1e-6] = 1.0
    # apply normalization to full X, then build datasets
    X = (X - feat_mean[None, :]) / feat_std[None, :]

    # save normalization stats
    np.savez(Path('tmp_out') / 'gaze_norm_stats.npz', mean=feat_mean, std=feat_std)

    train_ds = gaze_mlp.LandmarkGazeDataset(X[train_idx], Y[train_idx])
    val_ds = gaze_mlp.LandmarkGazeDataset(X[val_idx], Y[val_idx])
    
    # Optional: balance datasets via weighted sampling
    if getattr(args, 'balance_datasets', False):
        # Compute sample weights for the training set to achieve 60/40 ratio favoring MPIIGaze
        train_labels = [dataset_labels[i] for i in train_idx]
        mpii_count = sum(1 for lbl in train_labels if lbl == 'mpii')
        columbia_count = sum(1 for lbl in train_labels if lbl == 'columbia')
        print(f'Training set: {mpii_count} MPIIGaze, {columbia_count} Columbia')
        
        # Target: 60% MPIIGaze, 40% Columbia samples per epoch
        # Weight inversely to achieve desired ratio
        total = mpii_count + columbia_count
        mpii_target_weight = 0.6 / mpii_count if mpii_count > 0 else 0.0
        columbia_target_weight = 0.4 / columbia_count if columbia_count > 0 else 0.0
        
        weights = []
        for lbl in train_labels:
            if lbl == 'columbia':
                weights.append(columbia_target_weight)
            else:
                weights.append(mpii_target_weight)
        
        # Use WeightedRandomSampler (replacement=True for oversampling)
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
        print('Using weighted sampling to balance MPIIGaze and Columbia datasets')
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Support optional `--dropout` CLI flag; fallback to 0.2 if not provided.
    # Prepare debug overlay info: choose multiple annotation lines to visualize each epoch
    # Debug overlay selection: exactly one random Columbia and one random MPIIGaze line (if available).
    debug_ann_lines = []
    if len(ann_lines) > 0:
        import random
        columbia_lines = [l for l in ann_lines if 'columbia' in l.lower()]
        mpii_lines = [l for l in ann_lines if 'columbia' not in l.lower()]
        if columbia_lines:
            debug_ann_lines.append(random.choice(columbia_lines))
        if mpii_lines:
            debug_ann_lines.append(random.choice(mpii_lines))
        # Optional explicit debug image override: append if found (will create a third image)
        if getattr(args, 'debug_image', None):
            found = None
            for l in ann_lines:
                toks = l.split()
                if not toks:
                    continue
                imgp = toks[0]
                if imgp.endswith(args.debug_image) or Path(imgp).name == args.debug_image:
                    found = l
                    break
            if found and found not in debug_ann_lines:
                debug_ann_lines.append(found)
    debug_out_dir = Path(getattr(args, 'debug_out_dir', 'tmp_out/debug_epoch'))
    if len(debug_ann_lines) > 0:
        debug_out_dir.mkdir(parents=True, exist_ok=True)
        try:
            base_options = python.BaseOptions(model_asset_path=args.model_asset_path)
            debug_detector = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
            )
        except Exception as e:
            print('Failed to create debug face landmarker:', e)
            debug_detector = None
    else:
        debug_detector = None

    dropout_val = getattr(args, 'dropout', 0.15)
    model = gaze_mlp.SimpleMLP(in_dim, hidden=args.hidden, dropout=dropout_val)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Use ReduceLROnPlateau: reduce LR when validation plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    
    # direction loss uses cosine-style objective; confidence uses BCEWithLogits
    bce_loss = nn.BCEWithLogitsLoss()

    # Optional confidence training flags (backwards compatible)
    train_confidence = getattr(args, 'train_confidence', False)
    conf_weight = getattr(args, 'conf_weight', 1.0)

    best_val = float('inf')
    epochs_since_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
            if isinstance(out, tuple) or (hasattr(out, '__len__') and len(out) == 2):
                dir_raw, conf_logit = out
            else:
                dir_raw = out
                conf_logit = None
            # ensure batch-dim exists (some models/inputs can produce 1D outputs for single samples)
            if dir_raw.dim() == 1:
                dir_raw = dir_raw.unsqueeze(0)
            if yb.dim() == 1:
                yb = yb.unsqueeze(0)
            # normalize predicted directions
            pred_norm = torch.sqrt(torch.clamp((dir_raw ** 2).sum(dim=1, keepdim=True), min=1e-8))
            dir_unit = dir_raw / pred_norm
            # direction loss: 1 - dot(pred, target)
            dot = (dir_unit * yb).sum(dim=1)
            dir_loss = (1.0 - dot).mean()
            loss = dir_loss
            if train_confidence:
                # encourage confidence = 1 for training samples (simple proxy)
                conf_target = torch.ones((xb.size(0), 1), device=device)
                conf_loss = bce_loss(conf_logit, conf_target)
                loss = loss + conf_weight * conf_loss
            loss.backward()
            opt.step()
            sum_loss += float(loss.item()) * xb.size(0)
        train_loss = sum_loss / len(train_ds)

        # val - compute mean angle (deg) and optional conf metrics
        model.eval()
        sum_ang = 0.0
        ang_count = 0
        sum_loss = 0.0
        # Track per-dataset metrics if balancing is enabled
        if getattr(args, 'balance_datasets', False):
            val_labels = [dataset_labels[i] for i in val_idx]
            mpii_ang_sum = 0.0
            mpii_ang_count = 0
            columbia_ang_sum = 0.0
            columbia_ang_count = 0
        with torch.no_grad():
            for batch_i, (xb, yb) in enumerate(val_loader):
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                if isinstance(out, tuple) or (hasattr(out, '__len__') and len(out) == 2):
                    dir_raw, conf_logit = out
                else:
                    dir_raw = out
                    conf_logit = None
                if dir_raw.dim() == 1:
                    dir_raw = dir_raw.unsqueeze(0)
                if yb.dim() == 1:
                    yb = yb.unsqueeze(0)
                pred_norm = torch.sqrt(torch.clamp((dir_raw ** 2).sum(dim=1, keepdim=True), min=1e-8))
                dir_unit = dir_raw / pred_norm
                dot = (dir_unit * yb).sum(dim=1).clamp(-1.0, 1.0)
                ang = torch.acos(dot) * (180.0 / math.pi)
                sum_ang += float(ang.sum().item())
                ang_count += ang.numel()
                # Track per-dataset angles
                if getattr(args, 'balance_datasets', False):
                    batch_start = batch_i * args.batch_size
                    for sample_i, angle_val in enumerate(ang):
                        global_idx = batch_start + sample_i
                        if global_idx < len(val_labels):
                            if val_labels[global_idx] == 'columbia':
                                columbia_ang_sum += float(angle_val.item())
                                columbia_ang_count += 1
                            else:
                                mpii_ang_sum += float(angle_val.item())
                                mpii_ang_count += 1
                # loss for logging
                dir_loss = (1.0 - dot).mean()
                loss = dir_loss
                if train_confidence:
                    conf_target = torch.ones((xb.size(0), 1), device=device)
                    conf_loss = bce_loss(conf_logit, conf_target)
                    loss = loss + conf_weight * conf_loss
                sum_loss += float(loss.item()) * xb.size(0)
        val_loss = sum_loss / len(val_ds)
        mean_angle = sum_ang / max(1, ang_count)
        
        # Print per-dataset metrics if balancing enabled
        mpii_mean = float('nan')
        columbia_mean = float('nan')
        if getattr(args, 'balance_datasets', False):
            mpii_mean = mpii_ang_sum / max(1, mpii_ang_count)
            columbia_mean = columbia_ang_sum / max(1, columbia_ang_count)
            print(f'Epoch {epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_mean_angle_deg={mean_angle:.3f} | MPIIGaze={mpii_mean:.3f}° Columbia={columbia_mean:.3f}°')
        else:
            print(f'Epoch {epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_mean_angle_deg={mean_angle:.3f}')

        # Append CSV row if enabled
        if log_csv_path:
            try:
                from csv import writer as _csv_writer
                with open(log_csv_path, 'a', newline='') as f:
                    w = _csv_writer(f)
                    # current learning rate (first param group)
                    current_lr = opt.param_groups[0]['lr'] if opt.param_groups else float('nan')
                    w.writerow([epoch, f'{train_loss:.6f}', f'{val_loss:.6f}', f'{mean_angle:.6f}', f'{mpii_mean:.6f}', f'{columbia_mean:.6f}', f'{current_lr:.6e}'])
            except Exception as e:
                print('Failed to write CSV row:', e)
        
        # Step the learning rate scheduler based on validation loss
        scheduler.step(val_loss)
        
        if val_loss < best_val:
            best_val = val_loss
            gaze_mlp.save_model(model, args.out_model)
            print('Saved best model to', args.out_model)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve > getattr(args, 'patience', 0):
                print(f'No improvement in validation loss for {epochs_since_improve} epochs (patience={args.patience}). Early stopping.')
                break

        # Save debug overlays for this epoch for each selected debug annotation line
        if len(debug_ann_lines) > 0 and debug_detector is not None:
            for di, debug_ann_line in enumerate(debug_ann_lines, start=1):
                out_path = debug_out_dir / f'epoch_{epoch:02d}_img{di:02d}.jpg'
                try:
                    one_image_debug.save_debug_overlay_from_annotation(
                        debug_ann_line,
                        debug_detector,
                        model=model,
                        feat_mean=feat_mean,
                        feat_std=feat_std,
                        device=device,
                        out_path=out_path,
                        img_index=di,
                    )
                    print('Wrote debug overlay for epoch', epoch, 'image', di, 'to', out_path)
                except Exception as e:
                    print('Failed to write debug overlay for epoch', epoch, 'image', di, e)

def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--data-file', help='path to a p10.txt annotation file (legacy)')
    group.add_argument('--data-files', nargs='+', help='one or more p10.txt files or directories containing p_*.txt files')
    p.add_argument('--model-asset-path', default='./face_landmarker.task', help='mediapipe face_landmarker.task path')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--val-split', type=float, default=0.2, help='fraction of data to hold out for validation (0.0-0.5)')
    p.add_argument('--dry-run', action='store_true', help='only build dataset and print counts, do not train')
    p.add_argument('--debug-image', default=None, help='image filename (or suffix) from annotations to save debug overlays for each epoch')
    p.add_argument('--debug-out-dir', default='tmp_out/debug_epoch', help='directory to save per-epoch debug overlay images')
    p.add_argument('--debug-images-count', type=int, default=3, help='number of random debug images to save per epoch')
    p.add_argument('--cache-landmarks', action='store_true', help='cache mediapipe landmarks per-image under tmp_out/landmarks to speed repeated runs')
    p.add_argument('--augment-rotations', default=None, help='comma-separated list of in-plane rotation angles (degrees) to augment, e.g. "90,180,270"')
    p.add_argument('--augment-scales', default=None, help='comma-separated scale factors to simulate zoom-out, e.g. "0.8,0.6"')
    p.add_argument('--augment-scale-pivot', default='center', choices=['center', 'face'], help='pivot for scale augmentation: "center" or "face"')
    p.add_argument('--out-model', default='tmp_out/gaze_mlp.pt')
    p.add_argument('--patience', type=int, default=10, help='early stopping patience (epochs with no val_loss improvement)')
    p.add_argument('--balance-datasets', action='store_true', help='use weighted sampling to balance MPIIGaze and Columbia dataset influence during training')
    args = p.parse_args()
    Path('tmp_out').mkdir(exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
