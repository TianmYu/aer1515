import argparse
import random
from pathlib import Path
import numpy as np
import json
import bisect

from data_utils import MultimodalWindowDataset


def find_npz_files(data_dir: Path):
    return sorted([str(p) for p in data_dir.rglob('*.npz')])


def map_idx_to_file_and_start(dataset: MultimodalWindowDataset, idx: int):
    # replicate dataset indexing math to recover file path and start frame
    if idx < 0:
        idx = len(dataset) + idx
    file_idx = bisect.bisect_right(dataset.cum_counts, idx) - 1
    if file_idx < 0:
        raise IndexError(idx)
    local_idx = idx - dataset.cum_counts[file_idx]
    start = local_idx * dataset.stride
    file_path = dataset.file_infos[file_idx]['file']
    return file_path, start


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='NPZ corpus root (where metadata.json lives)')
    p.add_argument('--seq_len', type=int, default=30)
    p.add_argument('--stride', type=int, default=15)
    p.add_argument('--n_samples', type=int, default=20)
    p.add_argument('--val_frac', type=float, default=0.1, help='fraction of files to reserve for validation')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_samples', action='store_true', help='save sampled windows to audit_samples.npz in data dir')
    args = p.parse_args()

    data_dir = Path(args.data)
    files = find_npz_files(data_dir)
    if len(files) == 0:
        print('No .npz files found under', data_dir)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    files_shuffled = files.copy()
    random.shuffle(files_shuffled)
    val_count = max(1, int(len(files_shuffled) * args.val_frac))
    val_files = files_shuffled[-val_count:]

    print(f'Found {len(files)} npz files; using {len(val_files)} for validation ({args.val_frac*100:.1f}%)')

    ds = MultimodalWindowDataset(root_dir=str(data_dir), seq_len=args.seq_len, stride=args.stride, files=val_files, backend='npz')

    total_windows = len(ds)
    print('Validation windows:', total_windows)
    if total_windows == 0:
        print('No windows in validation split (increase seq_len or use more files).')
        return

    # compute label distributions across validation
    count_label = 0
    count_future = 0
    for i in range(total_windows):
        s = ds[i]
        if int(s['label'].item()) == 1:
            count_label += 1
        if int(s['future_label'].item()) == 1:
            count_future += 1

    print('Label counts (label / future_label):', count_label, '/', count_future)
    print('Label fractions: label=', count_label/total_windows, ' future_label=', count_future/total_windows)

    # sample some windows for inspection
    n_samples = min(args.n_samples, total_windows)
    sample_idxs = sorted(np.random.choice(total_windows, size=n_samples, replace=False).tolist())
    samples = []
    print('\nSampled windows (index, file, start, label, future_label, traj_mean, pose_preview[0:6]):')
    for si in sample_idxs:
        file_path, start = map_idx_to_file_and_start(ds, si)
        s = ds[si]
        traj = s['traj'].numpy()
        pose = s['pose'].numpy()
        lbl = int(s['label'].item())
        fut = int(s['future_label'].item())
        traj_mean = traj.mean(axis=0) if traj.size else np.array([])
        pose_preview = pose.flatten()[:6].tolist() if pose.size else []
        print(f'{si:6d} | {Path(file_path).name:30s} | start={start:4d} | label={lbl} future={fut} | traj_mean={traj_mean} | pose_preview={pose_preview}')
        samples.append({'idx': int(si), 'file': file_path, 'start': int(start), 'label': lbl, 'future': fut, 'traj_mean': traj_mean, 'pose_preview': pose_preview})

    if args.save_samples:
        outp = data_dir / 'audit_samples.json'
        # convert numpy arrays to lists for json
        for s in samples:
            s['traj_mean'] = s['traj_mean'].tolist() if hasattr(s['traj_mean'], 'tolist') else []
        with open(outp, 'w') as fh:
            json.dump({'summary': {'total_files': len(files), 'val_files': len(val_files), 'val_windows': total_windows}, 'samples': samples}, fh, indent=2)
        print('Saved sampled summary to', str(outp))


if __name__ == '__main__':
    main()
