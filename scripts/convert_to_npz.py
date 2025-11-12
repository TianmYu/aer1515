"""Convert per-person processed CSVs into compact NPZ shards.

Each output NPZ will contain these arrays:
  - pose: float32 array (N_frames, F_pose)
  - traj: float32 array (N_frames, 2)
  - label: int8 array (N_frames,) -- interacting label per frame
  - future_label: int8 array (N_frames,) -- future_interaction per frame

This script is intentionally simple and robust. Use it when you want fast
multi-epoch training. NPZ files are compressed (np.savez_compressed) to save disk.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import json


def convert_file(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path)

    # pose columns: *_x and *_y
    pose_cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
    pose_cols = sorted(pose_cols)
    if len(pose_cols) == 0:
        pose = np.zeros((len(df), 0), dtype=np.float32)
    else:
        pose = df[pose_cols].fillna(0.0).to_numpy(dtype=np.float32)

    # trajectory: pelvis_x/pelvis_y or cart_x/cart_y if present
    traj = None
    if 'pelvis_x' in df.columns and 'pelvis_y' in df.columns:
        traj = df[['pelvis_x', 'pelvis_y']].fillna(0.0).to_numpy(dtype=np.float32)
    elif 'cart_x' in df.columns and 'cart_y' in df.columns:
        traj = df[['cart_x', 'cart_y']].fillna(0.0).to_numpy(dtype=np.float32)
    else:
        traj = np.zeros((len(df), 2), dtype=np.float32)

    # labels
    if 'interacting' in df.columns:
        label = df['interacting'].fillna(False).astype(np.int8).to_numpy()
    else:
        label = np.zeros((len(df),), dtype=np.int8)

    if 'future_interaction' in df.columns:
        future = df['future_interaction'].fillna(False).astype(np.int8).to_numpy()
    elif 'will_interact' in df.columns:
        future = df['will_interact'].fillna(False).astype(np.int8).to_numpy()
    else:
        future = np.zeros((len(df),), dtype=np.int8)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (csv_path.stem + '.npz')
    # If trajectory is all zeros and we have pose columns, derive a compact traj from pose
    traj_derived = 0
    if (traj.shape[1] == 2 and np.allclose(traj, 0.0)) and len(pose) > 0 and len(pose_cols) > 0:
        try:
            derived = compute_traj_from_pose(pose, pose_cols)
            if derived is not None and derived.shape[0] == traj.shape[0]:
                traj = derived.astype(np.float32)
                traj_derived = 1
        except Exception:
            # Derivation failed; fall back to zeros
            traj_derived = 0

    np.savez_compressed(str(out_path), pose=pose, traj=traj, label=label, future_label=future, traj_derived=np.int8(traj_derived))
    return out_path


def compute_traj_from_pose(pose_flat: np.ndarray, pose_cols: list, root_name: str = None, dt: float = 1.0, smooth_window: int = None):
    """
    Compute a compact trajectory from flattened pose columns.
    - pose_flat: (T, F) numpy array where columns correspond to pose_cols
    - pose_cols: list of column names (must include *_x and *_y for joints)
    - root_name: optional joint base name to use as root (e.g. 'pelvis'). If not given, try 'pelvis', else first joint.
    Returns: traj (T, 6) -> [x_rel, y_rel, vx, vy, speed, heading]
    """
    import numpy as _np

    if pose_flat is None or pose_flat.size == 0:
        return None

    # infer joint base names
    bases = []
    for c in pose_cols:
        if '_' in c:
            base = c.rsplit('_', 1)[0]
            bases.append(base)
    bases = sorted(list(dict.fromkeys(bases)))
    if len(bases) == 0:
        return None

    # build mapping base -> (xcol, ycol)
    coords = {}
    for b in bases:
        x = b + '_x'
        y = b + '_y'
        if x in pose_cols and y in pose_cols:
            coords[b] = (pose_cols.index(x), pose_cols.index(y))

    if len(coords) == 0:
        return None

    # choose root
    root = None
    if root_name and root_name in coords:
        root = root_name
    elif 'pelvis' in coords:
        root = 'pelvis'
    else:
        root = next(iter(coords.keys()))

    x_idx, y_idx = coords[root]

    T = pose_flat.shape[0]
    x = pose_flat[:, x_idx].astype(float)
    y = pose_flat[:, y_idx].astype(float)

    # interpolate NaNs
    def interp_nan(arr):
        arr = _np.array(arr, dtype=float)
        n = arr.shape[0]
        inds = _np.arange(n)
        mask = _np.isnan(arr)
        if mask.all():
            return _np.zeros_like(arr)
        if mask.any():
            good = ~mask
            arr[mask] = _np.interp(inds[mask], inds[good], arr[good])
        return arr

    x = interp_nan(x)
    y = interp_nan(y)

    # optional smoothing via simple moving average
    if smooth_window and smooth_window > 1:
        k = _np.ones(smooth_window) / float(smooth_window)
        x = _np.convolve(x, k, mode='same')
        y = _np.convolve(y, k, mode='same')

    pos_rel_x = x - x[0]
    pos_rel_y = y - y[0]

    vx = _np.zeros_like(pos_rel_x)
    vy = _np.zeros_like(pos_rel_y)
    if T > 1:
        vx[1:] = (pos_rel_x[1:] - pos_rel_x[:-1]) / float(dt)
        vy[1:] = (pos_rel_y[1:] - pos_rel_y[:-1]) / float(dt)

    speed = _np.sqrt(vx ** 2 + vy ** 2)
    heading = _np.arctan2(vy, vx)

    traj = _np.stack([pos_rel_x, pos_rel_y, vx, vy, speed, heading], axis=1)
    traj = _np.nan_to_num(traj).astype(np.float32)
    return traj


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', type=str, default='datasets/processed', help='Input dir with per-person CSVs')
    p.add_argument('--output', '-o', type=str, default='datasets/npz', help='Output dir for NPZ files')
    p.add_argument('--max-files', type=int, default=None, help='Max files to convert')
    p.add_argument('--derive-traj', dest='derive_traj', action='store_true', help='Derive trajectory from pose when missing or zero')
    p.add_argument('--no-derive-traj', dest='derive_traj', action='store_false', help='Do not derive trajectory from pose')
    p.set_defaults(derive_traj=True)
    p.add_argument('--write-metadata', dest='write_metadata', action='store_true', help='Write aggregated metadata (mean/std) for converted files')
    p.set_defaults(write_metadata=True)
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    csvs = list(inp.rglob('*.csv'))
    if args.max_files:
        csvs = csvs[: args.max_files]

    converted = 0
    # accumulators for global stats
    global_traj_sum = None
    global_traj_sumsq = None
    global_frames = 0
    file_entries = []
    for c in csvs:
        try:
            pth = convert_file(c, out)
            print('Wrote', pth)
            converted += 1
            # load npz to read back traj and derived flag
            arr = np.load(str(pth))
            traj_arr = arr['traj']
            traj_derived = bool(arr.get('traj_derived', 0))
            n = traj_arr.shape[0]
            # update global accumulators
            if args.write_metadata:
                if global_traj_sum is None:
                    global_traj_sum = np.zeros(traj_arr.shape[1], dtype=float)
                    global_traj_sumsq = np.zeros(traj_arr.shape[1], dtype=float)
                global_traj_sum += traj_arr.sum(axis=0)
                global_traj_sumsq += (traj_arr ** 2).sum(axis=0)
                global_frames += n
                file_entries.append({'file': pth.name, 'n_frames': int(n), 'traj_derived': int(traj_derived)})
        except Exception as e:
            print('Failed', c, e)

    print(f'Converted {converted}/{len(csvs)} files to {out}')

    # write aggregated metadata
    if args.write_metadata and converted > 0 and global_frames > 0 and global_traj_sum is not None:
        meta = {'files': file_entries, 'total_files': converted, 'total_frames': int(global_frames)}
        traj_mean = (global_traj_sum / float(global_frames)).tolist()
        traj_var = (global_traj_sumsq / float(global_frames) - np.array(traj_mean) ** 2).tolist()
        traj_std = np.sqrt(np.maximum(np.array(traj_var), 0.0)).tolist()
        meta['traj_mean'] = traj_mean
        meta['traj_std'] = traj_std
        meta_path = out / 'metadata.json'
        with open(meta_path, 'w') as fh:
            json.dump(meta, fh, indent=2)
        print('Wrote metadata to', meta_path)


if __name__ == '__main__':
    main()
