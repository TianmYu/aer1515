# Code Improvements Summary

## Overview
This document summarizes all improvements made to the codebase including bug fixes, architectural enhancements, and training improvements.

## Critical Bug Fixes

### 1. Velocity Unit Conversion (scripts/convert_dataset_approach.py)
**Issue**: Velocity values from Dataset-approach were in mm/s but not converted to m/s, causing ~1000x scale mismatch with trajectory data.

**Fix**: Added proper conversion when `--mm-to-m` flag is set:
```python
vel = vel / 1000.0 if args.mm_to_m else vel
```

**Impact**: This bug would have caused severe model performance issues due to feature scale mismatch.

### 2. Semantic Pose Mismatch
**Issue**: Original dataset has 72-dim keypoint pose (36 Kinect skeleton keypoints × 2 coords), while approach dataset has 4-dim motion features [velocity, mTheta, oTheta, head]. Both were feeding the same encoder.

**Fix**: Implemented multi-encoder architecture (see below).

## Architectural Enhancements

### Multi-Encoder Architecture (transformer_model.py)
**Motivation**: Different pose representations require different encoders.

**Implementation**:
- Added separate encoders:
  - `keypoint_enc`: TemporalEncoder for 72-dim Kinect skeleton keypoints
  - `motion_enc`: TemporalEncoder for 4-dim motion features
  - `pose_enc`: Legacy encoder (kept for backward compatibility)
  
- Added routing based on `pose_type` metadata:
  - `pose_type='keypoint'` → use `keypoint_enc`
  - `pose_type='motion'` → use `motion_enc`
  - `pose_type=None` → use legacy `pose_enc`

- Added corresponding projection layers: `proj_keypoint`, `proj_motion`

**Benefits**:
- Each pose type gets specialized processing
- Can train on combined datasets with heterogeneous pose representations
- Backward compatible with existing models

### Trajectory-Only Mode (scripts/convert_dataset_approach.py)
**Motivation**: Allow model to work with trajectory data only when pose is unreliable or unavailable.

**Implementation**:
- Added `--traj-only` flag to converter
- When set, creates empty pose array: `pose = np.zeros((N, 0), dtype=np.float32)`
- Saves `pose_type=None` in metadata

**Benefits**:
- Can evaluate trajectory-only baseline
- More robust to pose estimation failures
- Useful for deployment scenarios with limited sensor data

### Pose Type Metadata (data_utils.py)
**Implementation**:
- Modified `_read_window()` to return pose_type as 5th element
- Added automatic pose_type inference:
  - From NPZ metadata (if saved during conversion)
  - From pose shape: 4-dim → 'motion', 72-dim → 'keypoint'
  - From CSV path (always → 'keypoint')
- Updated `collate_fn` to pass pose_type to model

**Benefits**:
- Automatic routing to correct encoder
- Backward compatible with old NPZ files (infers from shape)
- Enables mixed-dataset training

## Training Improvements (scripts/train_and_eval.py)

### 1. Reproducibility
- Added `set_seed()` function for all random number generators
- Set PyTorch deterministic mode when CUDA is available
- Fixed random seed usage (was re-seeding in data split)

### 2. Hyperparameter Control
Added command-line arguments:
- `--lr`: Learning rate (default 1e-3)
- `--weight-decay`: AdamW weight decay (default 1e-4)
- `--grad-clip`: Gradient clipping norm (default 1.0)
- `--focal-gamma`: Focal loss gamma parameter (default 2.0)
- `--focal-alpha`: Focal loss alpha parameter (default 0.25)
- `--aux-weight`: Weight for auxiliary future prediction loss (default 1.0)
- `--d-model`, `--nhead`, `--num-layers`: Model architecture params

### 3. Learning Rate Scheduling
- Added `ReduceLROnPlateau` scheduler
- Monitors validation F1 score
- Reduces LR by 0.5 when F1 plateaus for 2 epochs
- Helps fine-tune model in later training stages

### 4. Gradient Clipping
- Added gradient norm clipping (default max_norm=1.0)
- Prevents exploding gradients
- Improves training stability

### 5. Improved Checkpointing
**Best Model Tracking**:
- Tracks best validation F1 score across all epochs
- Saves best model to `checkpoints/best_model.pt`
- Checkpoint includes: model state, optimizer state, scheduler state, epoch, best F1, args

**Save Options**:
- `--save-best-only`: Only save best model (saves disk space)
- Default: save best + all epoch checkpoints

**Checkpoint Contents**:
- Full training state for resume capability
- Hyperparameters for reproducibility

### 6. Early Stopping
- Added `--early-stop-patience` argument (default 0 = disabled)
- Stops training if validation F1 doesn't improve for N epochs
- Prevents overfitting and saves compute time

### 7. Better Loss Computation
**Focal Loss**:
- Moved focal loss to standalone function for clarity
- Made gamma and alpha configurable
- Consistent implementation for training and validation

**Auxiliary Loss Weighting**:
- Added `--aux-weight` to control future prediction loss contribution
- Default 1.0 (equal weighting with main loss)
- Allows experimentation with multi-task learning balance

### 8. Enhanced Metrics & Logging
**Training Metrics**:
- Track loss components separately: main loss, aux loss, total loss
- Compute F1, precision, recall, accuracy on training set each epoch
- Better visibility into model learning

**Validation Metrics**:
- Same comprehensive metrics as training
- Printed after each epoch
- LR changes logged automatically by scheduler

**Final Evaluation**:
- Loads best model for final evaluation
- Prints comprehensive metrics table
- Ensures reported performance is from best checkpoint

### 9. Performance Optimizations
- Added `pin_memory=True` for DataLoader when using CUDA
- Enables faster host-to-device transfer
- Minor speedup but good practice

### 10. Code Organization
**Focal Loss Function**:
- Moved to module-level function (was nested, duplicated)
- Added proper docstring
- Made configurable with alpha parameter

**Checkpoint Directory**:
- Ensured creation with `os.makedirs(exist_ok=True)`
- Moved before training loop

## Code Cleanup

### Archived Redundant Files
Moved to `archive/` directory:
- `train.py` (superseded by `scripts/train_and_eval.py`)
- `inspect_dataset.py`, `inspect_dataset_run.py`, `inspect_dataverse.py` (superseded by `scripts/inspect_npz.py`)
- `label_stats.py` (one-off analysis)
- `scripts/audit_npz.py` (superseded by `scripts/audit_windows.py`)

### Kept Active Files
- `scripts/train_and_eval.py`: Main training harness
- `scripts/inspect_npz.py`: NPZ file inspection
- `scripts/audit_windows.py`: Window-level data auditing
- `pretrain.py`: Reserved for future self-supervised pretraining

## Next Steps

### 1. Re-convert Approach Dataset
The old approach dataset has incorrect velocity units. Must delete and recreate:

```bash
# Delete old data
Remove-Item -Recurse -Force datasets/npz_approach
Remove-Item -Recurse -Force datasets/npz_approach_h4s

# Re-convert with fixes
python scripts/convert_dataset_approach.py `
    --input datasets/Dataset-approach/Dataset `
    --output datasets/npz_approach `
    --mm-to-m `
    --traj-only

# Re-label with H=4s
python scripts/relabel_npz_future.py `
    --input datasets/npz_approach `
    --output datasets/npz_approach_h4s `
    --horizon 4.0
```

### 2. Smoke Test
Quick validation run with corrected data:

```bash
python scripts/train_and_eval.py `
    --data datasets/npz_h4s `
    --epochs 3 `
    --train-steps 50 `
    --batch 16 `
    --weighted-sampler `
    --class-weighted `
    --use-focal `
    --lr 1e-3 `
    --checkpoint-dir checkpoints/smoke_test
```

### 3. Full Training Run
After smoke test passes:

```bash
python scripts/train_and_eval.py `
    --data datasets/npz_h4s `
    --epochs 50 `
    --train-steps 500 `
    --batch 32 `
    --weighted-sampler `
    --class-weighted `
    --use-focal `
    --focal-gamma 2.0 `
    --focal-alpha 0.25 `
    --lr 1e-3 `
    --weight-decay 1e-4 `
    --grad-clip 1.0 `
    --aux-weight 1.0 `
    --d-model 128 `
    --nhead 8 `
    --num-layers 4 `
    --early-stop-patience 5 `
    --save-best-only `
    --checkpoint-dir checkpoints/full_run
```

### 4. Trajectory-Only Baseline
Test trajectory-only mode:

```bash
# First re-convert approach data with --traj-only
python scripts/convert_dataset_approach.py `
    --input datasets/Dataset-approach/Dataset `
    --output datasets/npz_approach_trajonly `
    --mm-to-m `
    --traj-only

python scripts/relabel_npz_future.py `
    --input datasets/npz_approach_trajonly `
    --output datasets/npz_approach_trajonly_h4s `
    --horizon 4.0

# Train
python scripts/train_and_eval.py `
    --data datasets/npz_approach_trajonly_h4s `
    --epochs 50 `
    --batch 32 `
    ...
```

## Summary of Benefits

1. **Correctness**: Fixed critical velocity unit bug
2. **Flexibility**: Multi-encoder architecture supports heterogeneous datasets
3. **Robustness**: Trajectory-only mode for degraded scenarios
4. **Reproducibility**: Seed control, deterministic mode, checkpoint args
5. **Training Quality**: LR scheduling, gradient clipping, early stopping
6. **Monitoring**: Comprehensive metrics, best model tracking
7. **Efficiency**: Faster convergence with improved training loop
8. **Maintainability**: Code cleanup, better organization, clear separation of concerns
