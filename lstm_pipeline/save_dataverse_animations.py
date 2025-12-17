#!/usr/bin/env python3
"""Create GIF animations for a few Dataverse trajectories from processed pickles.

Saves output GIFs to `tmp_out/` with names `dataverse_<pickle>_<idx>.gif`.
"""
import os
import glob
import pickle
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch
import facing_estimator as fe


def load_pickles(pkl_dir: Path):
    pkl_paths = sorted(glob.glob(str(pkl_dir / 'preproc_out_*.pkl')))
    out = []
    for p in pkl_paths:
        with open(p, 'rb') as f:
            try:
                data = pickle.load(f)
            except Exception:
                # try legacy class mapping
                from preprocess_vis_data import DataPointSet
                class DPSUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if name == 'DataPointSet' and module in {'__main__', 'preprocess_vis_data'}:
                            return DataPointSet
                        return super().find_class(module, name)
                with open(p, 'rb') as f2:
                    data = DPSUnpickler(f2).load()
        out.append((Path(p).name, data))
    return out


def make_animation_for_dp(dp, out_path, max_frames=300, fps=30):
    # dp: DataPointSet
    keys = list(dp.data_points.keys())
    seq_len = min(pts.shape[1] for pts in dp.data_points.values())
    n_frames = min(seq_len, max_frames)

    all_y = np.concatenate([pts[1, :n_frames] for pts in dp.data_points.values()])
    all_z = np.concatenate([pts[2, :n_frames] for pts in dp.data_points.values()])
    y_margin = (all_y.max() - all_y.min()) * 0.1 + 1e-3
    z_margin = (all_z.max() - all_z.min()) * 0.1 + 1e-3

    fig, ax = plt.subplots(figsize=(5, 5))
    scat = ax.scatter([], [], c='steelblue')
    ax.set_xlim(all_y.min() - y_margin, all_y.max() + y_margin)
    ax.set_ylim(all_z.min() - z_margin, all_z.max() + z_margin)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('normalized y')
    ax.set_ylabel('normalized z')
    ax.set_aspect('equal')

    # compute facing estimator results (yaw per frame)
    yaw_arr = None
    facing_arr = None
    # prefer precomputed facing info stored on the DataPointSet
    if hasattr(dp, 'facing') and dp.facing is not None:
        try:
            yaw_arr = np.asarray(dp.facing.get('yaw_deg', []), dtype=np.float32)
            facing_arr = np.asarray(dp.facing.get('facing', []), dtype=np.int8)
        except Exception:
            yaw_arr = None
            facing_arr = None
    if yaw_arr is None:
        try:
            res = fe.estimate_facing_from_dptset(dp, smooth=True)
            yaw_arr = np.asarray(res.get('yaw_deg', []), dtype=np.float32)
            facing_arr = np.asarray(res.get('facing', []), dtype=np.int8)
        except Exception:
            yaw_arr = np.zeros(n_frames, dtype=np.float32)
            facing_arr = np.zeros(n_frames, dtype=np.int8)

    # create a persistent arrow patch to indicate facing direction
    arrow = FancyArrowPatch((0, 0), (0, 0), color='red', mutation_scale=12, lw=2)
    ax.add_patch(arrow)

    def _get_dp_key(*keys):
        for k in keys:
            v = dp.data_points.get(k)
            if v is not None:
                return v
        return None

    def update(frame_idx):
        ys, zs = [], []
        for pts in dp.data_points.values():
            if frame_idx < pts.shape[1]:
                ys.append(pts[1, frame_idx])
                zs.append(pts[2, frame_idx])
        if len(ys) == 0:
            scat.set_offsets(np.empty((0, 2)))
        else:
            scat.set_offsets(np.column_stack([ys, zs]))
        # compute arrow anchor (mid shoulders if available, else nose if available)
        ox, oz = 0.0, 0.0
        left = _get_dp_key('shoulder_left', 'Left Shoulder')
        right = _get_dp_key('shoulder_right', 'Right Shoulder')
        nose = _get_dp_key('nose', 'Nose')
        if left is not None and right is not None and frame_idx < left.shape[1] and frame_idx < right.shape[1]:
            mid = 0.5 * (left[:2, frame_idx] + right[:2, frame_idx])
            ox, oz = float(mid[0]), float(mid[1])
        elif nose is not None and frame_idx < nose.shape[1]:
            ox, oz = float(nose[1, frame_idx]), float(nose[2, frame_idx])

        # yaw from estimator (degrees). If missing, default 0.
        yaw = float(yaw_arr[frame_idx]) if frame_idx < yaw_arr.size else 0.0
        ang = np.deg2rad(yaw)
        L = 0.2  # arrow length in plot units (tunable)
        dx = L * np.sin(ang)
        dz = L * np.cos(ang)
        # update persistent arrow endpoints
        try:
            arrow.set_positions((ox, oz), (ox + dx, oz + dz))
        except Exception:
            # older matplotlib may not have set_positions; fallback to removing/adding
            arrow.remove()
            new_arrow = FancyArrowPatch((ox, oz), (ox + dx, oz + dz), color='red', mutation_scale=12, lw=2)
            ax.add_patch(new_arrow)
        facing_flag = ''
        try:
            ff = int(facing_arr[frame_idx]) if frame_idx < facing_arr.size else 0
            # facing_arr is now binary {0,1}: 1 -> toward, 0 -> away/unknown
            if ff == 1:
                facing_flag = 'toward'
            else:
                facing_flag = 'away'
        except Exception:
            facing_flag = ''
        title = f"frame {frame_idx} / {n_frames} | yaw={yaw:.1f}° | facing={facing_flag}"
        ax.set_title(title)
        return scat,

    # interval (ms) matches fps for smoother playback in the saved GIF
    interval_ms = int(1000 / fps)
    ani = FuncAnimation(fig, update, frames=range(n_frames), interval=interval_ms, blit=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    ani.save(str(out_path), writer=writer)
    plt.close(fig)


def main():
    # prefer resampled 30hz folder if present
    p30 = Path('datasets/processed_30hz')
    pproc = Path('datasets/processed')
    if p30.exists():
        pkl_dir = p30
    else:
        pkl_dir = pproc

    pickles = load_pickles(pkl_dir)
    if not pickles:
        print('No pickles found in', pkl_dir)
        return

    # Dataverse are the first two pickles by convention
    dataverse_pickles = pickles[:2]

    out_dir = Path('tmp_out')
    out_dir.mkdir(exist_ok=True)

    created = []
    for name, data in dataverse_pickles:
        # data expected to be (robot_trajs, human_trajs)
        if not isinstance(data, (list, tuple)) or len(data) < 2:
            continue
        robot_trajs, human_trajs = data
        # filter reasonable trajectories
        human_trajs = [dp for dp in human_trajs if dp.data_points and min(pts.shape[1] for pts in dp.data_points.values()) >= 10]
        if not human_trajs:
            continue
        # pick up to 3 random trajs
        chosen = random.sample(human_trajs, min(3, len(human_trajs)))
        for i, dp in enumerate(chosen):
            out_path = out_dir / f"dataverse_{name.replace('.pkl','')}_{i}.gif"
            try:
                make_animation_for_dp(dp, out_path, max_frames=300, fps=10)
                created.append(out_path)
                print('Wrote', out_path)
            except Exception as e:
                print('Failed to create animation for', name, i, e)

    if not created:
        print('No animations created')
    else:
        print('Created animations:')
        for p in created:
            print('  ', p)


if __name__ == '__main__':
    main()
