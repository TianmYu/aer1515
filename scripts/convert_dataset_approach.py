"""Convert the Dataset-approach CSVs into per-person NPZ files

The script traverses the input root, reads CSVs (handles files where headers are
commented with '#'), groups rows by person id, and writes a per-person .npz with
arrays compatible with the project's MultimodalWindowDataset (keys: 'pose',
'traj', 'label'). It also writes a metadata.json containing traj mean/std so
the existing loader can normalize trajectories.

Usage example:
  python scripts/convert_dataset_approach.py --input datasets/Dataset-approach/Dataset \
      --output datasets/npz_approach --mm-to-m

This is non-destructive: output folder will be created and existing files not
overwritten unless --overwrite is set.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import json
import os


def read_csv_with_commented_header(p: Path, default_cols=None):
    # Find the header line (starts with '#time' or '#') and extract column names
    cols = None
    default = default_cols or ['time','idx','id','type','x','y','z','vel','mTheta','oTheta','head','uniqueID','options']
    with p.open('r', encoding='utf-8', errors='ignore') as fh:
        for ln in fh:
            ln = ln.strip()
            if ln.startswith('#') and (ln.lower().startswith('#time') or ',' in ln):
                # strip leading '#' and whitespace
                cand = ln.lstrip('#').strip()
                # simple split on comma to get column names
                cols = [c.strip() for c in cand.split(',') if c.strip()]
                break

    if cols is None:
        cols = default

    # Use pandas to read file, skipping comment lines. Since header line itself
    # is commented we supply names=cols and header=None so data rows map to names.
    try:
        df = pd.read_csv(p, comment='#', header=None, names=cols, low_memory=False)
    except Exception:
        # fallback: try a basic read
        df = pd.read_csv(p, low_memory=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Dataset-approach root folder containing CSVs')
    parser.add_argument('--output', required=True, help='Output NPZ root (will be created)')
    parser.add_argument('--mm-to-m', action='store_true', help='Convert millimeters to meters for traj and velocity')
    parser.add_argument('--traj-only', action='store_true', help='Skip pose features and save trajectory-only (has_pose=False)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing npz files')
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise RuntimeError('Input path does not exist: ' + str(inp))
    out.mkdir(parents=True, exist_ok=True)

    csvs = list(inp.rglob('*.csv'))
    if len(csvs) == 0:
        print('No CSV files found under', inp)
        return

    created = 0
    traj_accum = []
    npz_paths = []

    for csv in sorted(csvs):
        # coerce to Path in case glob returned strings
        csv = Path(csv)
        # determine folder label by looking for parent folder names
        # csv.parts yields strings; normalize to lower-case names for folder checks
        parts = [str(p).lower() for p in csv.parts]
        if any('intentiontointeract' in p for p in parts):
            session_label = 1
        elif any('otherdistinctiveintention' in p for p in parts):
            session_label = 0
        else:
            # default: treat as negative unless located under known positive folder
            session_label = 0

        df = read_csv_with_commented_header(csv)
        if 'id' not in df.columns:
            # try 'uniqueID'
            if 'uniqueID' in df.columns:
                df = df.rename(columns={'uniqueID': 'id'})
            else:
                # can't split per-person; skip
                print('Skipping (no id column):', csv)
                continue

        # ensure time column present
        if 'time' in df.columns:
            df = df.sort_values('time')

        for pid, g in df.groupby('id'):
            # drop rows with NaN x/y
            if 'x' not in g.columns or 'y' not in g.columns:
                # skip if no trajectory
                continue
            g = g[['time','x','y','z','vel','mTheta','oTheta','head']].copy()
            # coerce numeric
            for c in ['x','y','z','vel','mTheta','oTheta','head']:
                if c in g.columns:
                    g[c] = pd.to_numeric(g[c], errors='coerce').fillna(0.0)
                else:
                    g[c] = 0.0

            if len(g) == 0:
                continue

            # build arrays
            if args.traj_only:
                # Skip pose features entirely for trajectory-only mode
                pose = np.zeros((len(g), 0), dtype=np.float32)
                pose_type = None
            else:
                # pose features: [vel, mTheta, oTheta, head]
                # FIX: convert velocity from mm/s to m/s if requested
                vel = g['vel'].to_numpy(dtype=np.float32)
                if args.mm_to_m:
                    vel = vel / 1000.0
                pose = np.column_stack([
                    vel,
                    g['mTheta'].to_numpy(dtype=np.float32),
                    g['oTheta'].to_numpy(dtype=np.float32),
                    g['head'].to_numpy(dtype=np.float32)
                ])
                pose_type = 'motion'

            # traj: use x,y -> convert mm->m if requested
            traj = g[['x','y']].to_numpy(dtype=np.float32)
            if args.mm_to_m:
                traj = traj / 1000.0

            # label per-frame: session-level label
            labels = np.full((len(g),), session_label, dtype=np.uint8)

            # output path: preserve csv parent structure under output
            rel = csv.relative_to(inp)
            out_dir = out / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = csv.stem + f'_pid{pid}.npz'
            out_p = out_dir / stem
            if out_p.exists() and not args.overwrite:
                print('Skipping existing', out_p)
                continue

            # save npz with keys expected by loader: 'pose', 'traj', 'label'
            # Also save pose_type if available
            if args.traj_only or pose_type is None:
                np.savez_compressed(out_p, pose=pose, traj=traj, label=labels)
            else:
                np.savez_compressed(out_p, pose=pose, traj=traj, label=labels, pose_type=pose_type)
            npz_paths.append(out_p)
            created += 1
            traj_accum.append(traj)

    # compute global traj mean/std and write metadata.json
    if len(traj_accum) > 0:
        all_traj = np.vstack([t for t in traj_accum if t.size > 0])
        traj_mean = all_traj.mean(axis=0).tolist()
        traj_std = all_traj.std(axis=0).tolist()
    else:
        traj_mean = [0.0, 0.0]
        traj_std = [1.0, 1.0]

    meta = {
        'source': 'Dataset-approach',
        'total_files': len(npz_paths),
        'traj_mean': traj_mean,
        'traj_std': traj_std,
        'traj_only': args.traj_only,
        'pose_type': None if args.traj_only else 'motion',
    }
    with open(out / 'metadata.json', 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2)

    print(f'Created {created} per-person npz files under {out} (metadata written)')


if __name__ == '__main__':
    main()
