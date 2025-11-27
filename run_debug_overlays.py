#!/usr/bin/env python3
"""Generate debug overlays for a few Columbia images using the trained MLP.

Writes overlays to `tmp_out/debug_columbia/`.
"""
import os
from pathlib import Path
import numpy as np
import torch

import gaze_mlp
import one_image_debug
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def main():
    base = Path(__file__).resolve().parent
    ann = base / 'MPIIGazeSet' / 'columbia_p10.txt'
    model_path = base / 'tmp_out' / 'gaze_mlp_balanced_6040_200ep.pt'
    norm_path = base / 'tmp_out' / 'gaze_norm_stats.npz'
    out_dir = base.parent / 'tmp_out' / 'debug_columbia'
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ann.exists():
        print('Annotations not found:', ann)
        return 1
    if not model_path.exists():
        print('Model not found:', model_path)
        return 1
    if not norm_path.exists():
        print('Norm stats not found:', norm_path)
        return 1

    data = np.load(str(norm_path))
    mean = data['mean']
    std = data['std']
    in_dim = int(mean.shape[0])

    # load model
    model = gaze_mlp.load_model(str(model_path), in_dim=in_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # create mediapipe face landmarker for debug overlays
    base_options = python.BaseOptions(model_asset_path=str(base / 'face_landmarker.task'))
    try:
        detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        )
    except Exception as e:
        print('Failed to create face landmarker:', e)
        detector = None

    # pick a few annotation lines (first N distinct subjects)
    LINES = [l for l in ann.read_text().splitlines() if l.strip()]
    if not LINES:
        print('No lines in annotation file')
        return 1

    # choose up to 6 samples evenly spaced
    N = min(6, len(LINES))
    step = max(1, len(LINES) // N)
    picks = [LINES[i] for i in range(0, len(LINES), step)][:N]

    saved = []
    for i, line in enumerate(picks, start=1):
        out_path = out_dir / f'columbia_debug_{i:02d}.jpg'
        try:
            one_image_debug.save_debug_overlay_from_annotation(
                line,
                detector,
                model=model,
                feat_mean=mean,
                feat_std=std,
                device=device,
                out_path=out_path,
                img_index=i,
            )
            print('Wrote overlay:', out_path)
            saved.append(str(out_path))
        except Exception as e:
            print('Failed overlay for line:', line.split()[0], e)

    print('\nSaved overlays:')
    for s in saved:
        print(' -', s)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
