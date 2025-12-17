#!/usr/bin/env python3
"""Resample low-fps DataPointSet elements in preprocessed pickles up to 30Hz.

This script walks a directory of `preproc_out_*.pkl` files, loads their
contents (which may be a tuple/list of DataPointSet objects), and for any
DataPointSet with an `fps` lower than the target it calls
`DataPointSet.resample_to_fps(target_fps)` which performs interpolation.

By default it writes outputs to `datasets/processed_30hz` to avoid overwriting
the originals.
"""
import argparse
import glob
import os
import pickle
from pathlib import Path
import sys


def main():
    # Simple configuration (no CLI): set to True to resample Dataverse to 30Hz
    RESAMPLE_DATAVERSE = True
    IN_DIR = Path('datasets/processed')
    OUT_DIR = Path('datasets/processed_30hz')
    TARGET_FPS = 30.0
    FORCE = False

    # Make pipeline modules importable
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    # Import helper from preprocess_vis_data
    try:
        from preprocess_vis_data import resample_datapoints
        from preprocess_vis_data import DataPointSet
    except Exception as e:
        print('Error importing from preprocess_vis_data:', e)
        raise

    class DPSUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'DataPointSet' and module in {'__main__', 'preprocess_vis_data'}:
                return DataPointSet
            return super().find_class(module, name)

    in_dir = IN_DIR
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_paths = sorted(glob.glob(str(in_dir / 'preproc_out_*.pkl')))
    print(f'Found {len(pkl_paths)} pickles in {in_dir}')

    # Import the reusable helper from preprocess_vis_data
    try:
        from preprocess_vis_data import resample_datapoints
    except Exception as e:
        print('Error importing resample_datapoints from preprocess_vis_data:', e)
        raise

    total_processed = 0
    total_resampled = 0

    for p in pkl_paths:
        total_processed += 1
        with open(p, 'rb') as f:
            try:
                data = DPSUnpickler(f).load()
            except Exception as e:
                print(f'Failed to unpickle {p}:', e)
                continue

        # data may be a tuple like (robot_trajs, human_trajs) or list
        if RESAMPLE_DATAVERSE:
            changed = resample_datapoints(data, TARGET_FPS, force=FORCE)
        else:
            changed = False

        if changed:
            total_resampled += 1
            out_path = out_dir / Path(p).name
            with open(out_path, 'wb') as f_out:
                pickle.dump(data, f_out)
            print(f'Wrote resampled pickle to {out_path}')
        else:
            # copy original file to out_dir to keep directory consistent
            out_path = out_dir / Path(p).name
            with open(p, 'rb') as f_in, open(out_path, 'wb') as f_out:
                f_out.write(f_in.read())

    print(f'Done. Processed: {total_processed}, Resampled files: {total_resampled}.')


if __name__ == '__main__':
    main()
