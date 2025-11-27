from pathlib import Path
import sys
import numpy as np
import math

# Ensure local package imports from this folder work
sys.path.insert(0, str(Path(__file__).resolve().parent))

import one_image_debug as oid
import gaze_mlp

# mediapipe imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import torch

# choose an annotation file and the first line
base_dir = Path(__file__).resolve().parent / 'MPIIGazeSet'
ann_files = sorted([p for p in base_dir.rglob('p*.txt')])
if len(ann_files) == 0:
    raise SystemExit(f'no per-subject annotation files found under {base_dir}')
ann_path = ann_files[0]
print('Using annotation file:', ann_path)
lines = [l.strip() for l in ann_path.read_text().splitlines() if l.strip()]
if len(lines) == 0:
    raise SystemExit('no annotation lines')
line = lines[0]
print('Using annotation line:', line.split()[0])

# create detector
model_asset = str((Path(__file__).resolve().parent / 'face_landmarker.task').resolve())
base_options = python.BaseOptions(model_asset_path=model_asset)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# parse GT from annotation
toks = line.split()
fc = np.array([float(toks[21]), float(toks[22]), float(toks[23])])
gt = np.array([float(toks[24]), float(toks[25]), float(toks[26])])
vec = gt - fc
norm = np.linalg.norm(vec)
unit_gt = vec / (norm if norm > 1e-12 else 1e-12)
print('\nGROUND TRUTH')
print('fc:', fc)
print('gt:', gt)
print('vec:', vec)
print('unit_gt:', unit_gt)
# yaw/pitch deg
def vec_to_yaw_pitch_deg(v):
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    # If gaze points toward camera (negative z), negate for yaw/pitch calc
    if z < 0:
        x, y, z = -x, -y, -z
    yaw = math.degrees(math.atan2(x, z))
    denom = math.sqrt(max(0.0, x*x + z*z))
    pitch = 0.0 if denom <= 1e-12 else math.degrees(math.atan2(-y, denom))
    return yaw, pitch

yaw_gt, pitch_gt = vec_to_yaw_pitch_deg(unit_gt)
print(f'GT yaw:{yaw_gt:+.3f} pitch:{pitch_gt:+.3f}')

# detect landmarks for the image
# image paths in annotations are relative to the annotation file directory
img_path = (Path(__file__).resolve().parent / 'MPIIGazeSet' / 'p10' / toks[0])
if not img_path.exists():
    raise SystemExit(f'image not found: {img_path}')
mp_img = None
try:
    mp_img = python.Image.create_from_file(str(img_path))
except Exception:
    # some mediapipe versions require mp.Image import path; try alternative
    try:
        import mediapipe as mp
        mp_img = mp.Image.create_from_file(str(img_path))
    except Exception as e:
        print('Failed to create mediapipe image:', e)
        raise
res = detector.detect(mp_img)
if not hasattr(res, 'face_landmarks') or len(res.face_landmarks) == 0:
    raise SystemExit('No face detected for diagnostic')
face_landmarks = res.face_landmarks[0]

# build feature vector
lm_pts = [(float(lm.x), float(lm.y), float(lm.z)) for lm in face_landmarks]
feat = gaze_mlp.landmarks_to_feature_vector(lm_pts)
in_dim = feat.shape[0]
print('\nLandmark feature dim =', in_dim)

# attempt to load normalization stats
mean_path = Path('tmp_out/gaze_norm_stats.npz')
if mean_path.exists():
    stats = np.load(mean_path)
    feat_mean = stats['mean']
    feat_std = stats['std']
    print('Loaded feat mean/std from', mean_path)
else:
    feat_mean = None
    feat_std = None
    print('No feat mean/std found; predictions will be skipped if model requires them')

# attempt to load model
candidate_models = [Path('tmp_out/gaze_mlp_improved.pt'), Path('tmp_out/gaze_mlp_aug_100ep.pt'), Path('tmp_out/gaze_mlp_fresh_20ep.pt'), Path('tmp_out/gaze_mlp_combined_100ep.pt')]
model_path = None
for p in candidate_models:
    if p.exists():
        model_path = p
        break

model = None
if model_path is not None and feat_mean is not None and feat_std is not None:
    try:
        model = gaze_mlp.load_model(str(model_path), in_dim=in_dim)
        model.eval()
        print('Loaded model from', model_path)
    except Exception as e:
        print('Failed to load model:', e)
        model = None
else:
    print('Model not loaded (missing file or norm stats).')

# If model loaded, compute prediction numerically
if model is not None:
    try:
        feat_norm = (feat - feat_mean) / feat_std
        x_in = torch.from_numpy(feat_norm.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            outp = model(x_in)
        if isinstance(outp, tuple) or (hasattr(outp, '__len__') and len(outp) == 2):
            dir_raw_pred = outp[0]
        else:
            dir_raw_pred = outp
        if dir_raw_pred.dim() == 1:
            dir_raw_pred = dir_raw_pred.unsqueeze(0)
        pred_norm = torch.sqrt(torch.clamp((dir_raw_pred ** 2).sum(dim=1, keepdim=True), min=1e-8))
        dir_unit_pred = (dir_raw_pred / pred_norm).cpu().numpy().reshape(-1)
        print('\nPREDICTION')
        print('dir_raw_pred:', dir_raw_pred.cpu().numpy().reshape(-1))
        print('dir_unit_pred:', dir_unit_pred)
        yaw_p, pitch_p = vec_to_yaw_pitch_deg(dir_unit_pred)
        print(f'PRED yaw:{yaw_p:+.3f} pitch:{pitch_p:+.3f}')
    except Exception as e:
        print('Prediction failed:', e)

# save overlay using the library function (it will draw both GT and prediction if model provided)
outp = Path('tmp_out/diag_overlay.jpg')
try:
    # replace the relative image path in the annotation line with the absolute path
    toks2 = line.split()
    toks2[0] = str(img_path)
    line_abs = ' '.join(toks2)
    saved = oid.save_debug_overlay_from_annotation(line_abs, detector, model=model, feat_mean=feat_mean, feat_std=feat_std, device='cpu', out_path=outp, img_index=1)
    print('\nWrote overlay to', saved)
except Exception as e:
    print('Overlay creation failed:', e)

print('\nDiagnostic complete.')
