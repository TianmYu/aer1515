import argparse
from pathlib import Path
import numpy as np
import os


def compute_future_label(interacting_arr: np.ndarray, horizon_frames: int) -> np.ndarray:
    # interacting_arr: boolean array per frame indicating interaction at that frame
    n = interacting_arr.shape[0]
    fut = np.zeros_like(interacting_arr, dtype=np.uint8)
    if n == 0:
        return fut
    # For each frame i, fut[i] = 1 if any interacting[j] for j in (i+1 .. i+horizon_frames)
    # We can compute using convolution-like cumulative sum trick for speed
    # Compute exclusive moving sum of next horizon_frames
    cumsum = np.concatenate([[0], np.cumsum(interacting_arr.astype(np.int32))])
    # for frame i, sum over (i+1 .. i+h) == cumsum[min(n, i+h+1)] - cumsum[i+1]
    for i in range(n):
        end = min(n, i + horizon_frames + 1)
        s = cumsum[end] - cumsum[i + 1]
        fut[i] = 1 if s > 0 else 0
    return fut


def relabel_file(inp_path: Path, out_path: Path, horizon_frames: int):
    try:
        arr = np.load(inp_path)
    except Exception as e:
        print(f'Failed to load {inp_path}: {e}')
        return False

    # Determine interacting array from common keys
    if 'future_label' in arr:
        # If future_label exists, assume it's okay but recompute based on 'label' if available
        if 'label' in arr:
            interacting = arr['label'].astype(bool)
        elif 'interacting' in arr:
            interacting = arr['interacting'].astype(bool)
        else:
            # nothing to base on; keep existing future_label
            future = arr['future_label'].astype(np.uint8)
            # write copy
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, **{k: arr[k] for k in arr.files}, future_label=future)
            return True
    else:
        if 'label' in arr:
            interacting = arr['label'].astype(bool)
        elif 'interacting' in arr:
            interacting = arr['interacting'].astype(bool)
        else:
            # no interacting info; set zeros
            interacting = np.zeros(arr['pose'].shape[0], dtype=bool) if 'pose' in arr else np.zeros(0, dtype=bool)

    future = compute_future_label(interacting, horizon_frames)

    # Write a new npz: copy all arrays and add/overwrite future_label
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {k: arr[k] for k in arr.files}
    save_dict['future_label'] = future
    np.savez_compressed(out_path, **save_dict)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Input NPZ folder root')
    p.add_argument('--output', required=True, help='Output NPZ folder root (will be created)')
    p.add_argument('--horizon', type=float, default=4.0, help='Horizon in seconds (default 4s)')
    p.add_argument('--sample-rate', type=float, default=5.0, help='Frames per second in dataset (default 5 Hz)')
    p.add_argument('--dry-run', action='store_true', help='Only report files, do not write')
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise RuntimeError('Input path does not exist: ' + str(inp))

    horizon_frames = int(round(args.horizon * args.sample_rate))
    print(f'Relabeling NPZ files from {inp} -> {out} with horizon {args.horizon}s ({horizon_frames} frames)')

    npz_files = sorted([p for p in inp.rglob('*.npz')])
    if len(npz_files) == 0:
        print('No npz files found under', inp)
        return

    total = 0
    succeeded = 0
    for pth in npz_files:
        rel = pth.relative_to(inp)
        out_p = out / rel
        if args.dry_run:
            print('Would relabel', pth, '->', out_p)
            total += 1
            continue
        ok = relabel_file(pth, out_p, horizon_frames)
        total += 1
        if ok:
            succeeded += 1

    print(f'Processed {total} files, succeeded {succeeded}')


if __name__ == '__main__':
    main()
