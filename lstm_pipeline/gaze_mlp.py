import math
from typing import List, Tuple, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.utils.data as data
except Exception:
    torch = None


def landmarks_to_feature_vector(landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
    """Flatten landmarks (x,y,z) into a 1D numpy feature vector.

    Expects list of (x,y,z) in camera coordinates or normalized coords depending on source.
    Missing landmarks should be filled with zeros or nan-handled prior to use.
    """
    arr = np.asarray(landmarks, dtype=np.float32).reshape(-1)
    return arr


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 192, out_dim: int = 3, dropout: float = 0.15):
        """
        3-layer MLP with batch normalization and moderate dropout.
        Balanced architecture: not too deep, not too shallow.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(hidden // 2, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class LandmarkGazeDataset(data.Dataset):
    """Simple Dataset: features are flattened landmarks, targets are gaze vectors (gt - fc).

    Expects pre-built lists/arrays of features and targets.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0]
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_dataset_from_annotations(
    annotation_lines: List[str],
    face_landmark_extractor,  # callable that returns landmarks list for an image path
    rotations: Optional[List[int]] = None,
    scales: Optional[List[float]] = None,
    scale_pivot: str = 'center',  # 'center' or 'face'
    dataset_labels: Optional[List[str]] = None,  # optional labels to track dataset source
):
    """Given annotation lines from p10.txt and a callable that returns landmarks for an image,
    build arrays X (N x D) and y (N x 3). The face_landmark_extractor should accept an image path
    and return a list of (x,y,z) landmarks or None if face not found.
    
    If dataset_labels is provided, returns (X, Y, labels_out) where labels_out contains labels
    for successful samples only.
    """
    X = []
    Y = []
    labels_out = [] if dataset_labels is not None else None
    skipped_missing_file = 0
    skipped_no_face = 0
    for idx, line in enumerate(annotation_lines):
        parts = line.strip().split()
        if not parts or len(parts) != 28:
            continue
        rel_path = parts[0]
        vals = list(map(float, parts[15:27]))
        # canonical p10 mapping: parts[15:27] contains 12 floats where
        # vals[6:9] -> face center (fc), vals[9:12] -> gaze target (gt).
        # Compute gaze as gt - fc and normalize to unit vector.
        fc = np.array(vals[6:9], dtype=np.float32)
        gt = np.array(vals[9:12], dtype=np.float32)
        gaze_vec = gt - fc
        
        norm = np.linalg.norm(gaze_vec)
        if norm < 1e-6:
            # skip degenerate vectors
            skipped_no_face += 1
            continue
        gaze_unit = (gaze_vec / norm).astype(np.float32)
        
        # Apply dataset-specific coordinate transform (MPIIGaze uses opposite convention)
        import data_pipeline
        gaze_unit = data_pipeline.apply_dataset_coordinate_transform(gaze_unit, rel_path)
        landmarks = None
        try:
            landmarks = face_landmark_extractor(rel_path)
        except FileNotFoundError:
            skipped_missing_file += 1
            landmarks = None
        except Exception:
            # other extractor errors (e.g., image decode) treated as no-face
            landmarks = None

        if landmarks is None:
            # We can't determine whether it was a missing file or face not found
            # if the extractor doesn't raise FileNotFoundError. The extractor
            # may choose to return None for either case. Count conservatively.
            skipped_no_face += 1
            continue
        feat = landmarks_to_feature_vector(landmarks)
        X.append(feat)
        Y.append(gaze_unit)
        if labels_out is not None:
            labels_out.append(dataset_labels[idx])
        # apply optional in-plane rotations (image-plane augmentation)
        if rotations:
            for ang in rotations:
                a = int(ang) % 360
                if a == 0:
                    continue
                theta = math.radians(a)
                c = math.cos(theta)
                s = math.sin(theta)
                # rotate landmarks' (x,y) about image center (assumes normalized [0,1] coords)
                rlands = []
                for (lx, ly, lz) in landmarks:
                    dx = lx - 0.5
                    dy = ly - 0.5
                    rx = c * dx - s * dy + 0.5
                    ry = s * dx + c * dy + 0.5
                    rlands.append((float(rx), float(ry), float(lz)))
                rfeat = landmarks_to_feature_vector(rlands)
                # rotate gaze vector x,y (camera-frame) about z-axis
                gx, gy, gz = gaze_unit[0], gaze_unit[1], gaze_unit[2]
                rgx = c * gx - s * gy
                rgy = s * gx + c * gy
                rgaze = np.array([rgx, rgy, gz], dtype=np.float32)
                # renormalize to unit length to avoid numerical drift
                ng = np.linalg.norm(rgaze)
                if ng < 1e-8:
                    continue
                rgaze = (rgaze / ng).astype(np.float32)
                X.append(rfeat)
                Y.append(rgaze)
        # apply optional scaling augmentation to simulate zoom-out (landmarks get closer)
        if scales:
            # determine pivot point for scaling: image center or face center from annotation
            if scale_pivot not in ('center', 'face'):
                pivot = 'center'
            else:
                pivot = scale_pivot
            # if pivot is face and fc was provided in annotation, use it; otherwise use image center
            if pivot == 'face':
                pivot_x, pivot_y = float(fc[0]), float(fc[1])
            else:
                pivot_x, pivot_y = 0.5, 0.5
            for s in scales:
                try:
                    sf = float(s)
                except Exception:
                    continue
                if sf <= 0.0:
                    continue
                slands = []
                for (lx, ly, lz) in landmarks:
                    rx = pivot_x + (lx - pivot_x) * sf
                    ry = pivot_y + (ly - pivot_y) * sf
                    rz = float(lz) * float(sf)
                    slands.append((float(rx), float(ry), float(rz)))
                sfeat = landmarks_to_feature_vector(slands)
                # For scale augmentation we keep the gaze vector in camera frame unchanged
                # (because in-plane image scaling/padding doesn't change true 3D gaze).
                X.append(sfeat)
                Y.append(gaze_unit)

    if len(X) == 0:
        print(f'build_dataset_from_annotations: no samples collected; skipped_missing_file={skipped_missing_file} skipped_no_face={skipped_no_face}')
        empty_X = np.zeros((0, 0), dtype=np.float32)
        empty_Y = np.zeros((0, 3), dtype=np.float32)
        if labels_out is not None:
            return empty_X, empty_Y, []
        return empty_X, empty_Y
    X = np.vstack(X)
    Y = np.vstack(Y)
    print(f'build_dataset_from_annotations: collected {X.shape[0]} samples; skipped_missing_file={skipped_missing_file} skipped_no_face={skipped_no_face}')
    if labels_out is not None:
        return X, Y, labels_out
    return X, Y


def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def load_model(path: str, in_dim: int, hidden: int = 128, out_dim: int = 3) -> nn.Module:
    model = SimpleMLP(in_dim, hidden, out_dim)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model
