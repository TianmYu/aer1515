"""Simple facing-direction estimator using YOLO keypoints.

Functions here operate on DataPointSet-like `data_points` mappings where
each value is a numpy array of shape (C, N) (C=2 or 3 channels, N frames).

Primary API:
 - estimate_facing_from_dptset(dptset, frames=None, use_3d=True, smooth=False, alpha=0.2)

"""
from typing import Optional, Sequence, Tuple, Dict, Any
import numpy as np
import math


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * (a + b)


def _get_point_at_frame(arr: Optional[np.ndarray], idx: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 1:
        return a.copy()
    if a.ndim == 2:
        if idx < 0 or idx >= a.shape[1]:
            return None
        return a[:, idx].copy()
    return None


def estimate_facing_from_dptset(
    dptset,
    frames: Optional[Sequence[int]] = None,
    use_3d: bool = True,
    smooth: bool = False,
    alpha: float = 0.2,
    mode: str = "auto",  # 'auto'|'image2d'|'3d'
    debug: bool = False,
) -> Dict[str, Any]:
    """Estimate yaw (degrees) per frame and confidence.

    Args:
      dptset: object with attribute `data_points` mapping key -> np.ndarray (C, N)
      frames: iterable of frame indices to compute (None -> all frames available)
      use_3d: prefer z coordinate when available
      smooth: apply simple EMA smoothing over yaw sequence
      alpha: smoothing factor if `smooth` True

        Returns:
            dict with keys: `yaw_deg` (np.ndarray), `confidence` (np.ndarray), `method` (list),
            and `facing` (np.ndarray) where `1` means facing towards the camera, `-1` away, `0` unknown.
    """

    d = dptset.data_points
    # build case-insensitive key map so estimator works with different pickle formats
    key_map = {k.lower(): k for k in d.keys()}
    # determine number of frames by inspecting any available key
    any_key = next(iter(d.keys()))
    total_frames = d[any_key].shape[1]
    if frames is None:
        idxs = list(range(total_frames))
    else:
        idxs = [i for i in frames if 0 <= i < total_frames]

    yaws = []
    confs = []
    methods = []

    for i in idxs:
        # pick the best available nose/eye point (case-insensitive)
        def _kp(k):
            orig = key_map.get(k)
            return _get_point_at_frame(d.get(orig) if orig is not None else None, i)

        nose = _kp('nose')
        if nose is None:
            nose = _kp('eye_left')
        if nose is None:
            nose = _kp('eye_right')
        left_sh = _kp('shoulder_left')
        right_sh = _kp('shoulder_right')
        left_hip = _kp('hip_left')
        right_hip = _kp('hip_right')

        yaw = 0.0
        conf = 0.0
        method = 'unknown'

        if left_sh is not None and right_sh is not None and nose is not None:
            mid_sh = _mid(left_sh, right_sh)

            # Prefer an image-plane 2D method when coordinates are 2D or when
            # caller requests 'image2d' mode. Otherwise fall back to the original
            # 3D-style vector (shoulder-nose) computation.
            try_image2d = False
            if mode == "image2d":
                try_image2d = True
            elif mode == "3d":
                try_image2d = False
            else:  # auto
                try_image2d = (min(left_sh.shape[0], right_sh.shape[0], nose.shape[0]) == 2)

            if try_image2d:
                # 2D image-plane forward direction: perpendicular to shoulders
                sh_vec = np.asarray(right_sh)[:2] - np.asarray(left_sh)[:2]
                if np.linalg.norm(sh_vec) < 1e-6:
                    # degenerate shoulders -> fallback to PCA below
                    yaw = 0.0
                    conf = 0.0
                    method = 'image2d-failed'
                else:
                    perp = np.array([-sh_vec[1], sh_vec[0]])
                    # normalize perp for stability
                    perp_norm = perp / (np.linalg.norm(perp) + 1e-8)
                    # Resolve sign ambiguity using nose relative to mid-shoulder
                    try:
                        mid2 = np.asarray(mid_sh)[:2]
                        nose2 = np.asarray(nose)[:2]
                        nose_vec2 = nose2 - mid2
                        if np.dot(nose_vec2, perp_norm) < 0:
                            perp_norm = -perp_norm
                    except Exception:
                        # if nose/mid unavailable or malformed, skip sign-correction
                        pass
                    yaw = math.degrees(math.atan2(float(perp_norm[1]), float(perp_norm[0])))
                    # normalize yaw to [-180, 180]
                    yaw = ((yaw + 180.0) % 360.0) - 180.0
                    shoulder_width = float(np.linalg.norm(sh_vec))
                    # estimate subject scale from bbox of keypoints (2D)
                    pts_for_scale = []
                    for p in (left_sh, right_sh, left_hip, right_hip, nose):
                        if p is None:
                            continue
                        pa = np.asarray(p)
                        pts_for_scale.append(pa[:2])
                    if pts_for_scale:
                        P = np.stack(pts_for_scale, axis=0)
                        bb_min = P.min(axis=0)
                        bb_max = P.max(axis=0)
                        subject_scale = float(np.linalg.norm(bb_max - bb_min))
                    else:
                        subject_scale = float(max(shoulder_width, 1e-6))
                    expected_shoulder_frac = 0.25
                    conf = float(np.clip(shoulder_width / (expected_shoulder_frac * subject_scale + 1e-8), 0.0, 1.0))
                    method = 'image2d'
            else:
                # original depth/lateral style computation
                vec = nose - mid_sh
                # Coordinates are typically [x(depth), y(lateral), z(vertical)].
                depth = 0.0
                lateral = 0.0
                if vec.shape[0] >= 1:
                    depth = float(vec[0])
                if vec.shape[0] >= 2:
                    lateral = float(vec[1])
                # If depth is effectively zero (or missing) but a 3rd axis exists,
                # treat axis-2 as depth and axis-1 as lateral instead.
                if abs(depth) < 1e-6 and vec.shape[0] >= 3:
                    lateral = float(vec[1])
                    depth = float(vec[2])
                yaw = math.degrees(math.atan2(lateral, depth))
                # Compute shoulder width using the two most informative axes.
                if left_sh.shape[0] >= 3:
                    sh_vec = (right_sh[1:3] - left_sh[1:3])
                else:
                    sh_vec = (right_sh[:2] - left_sh[:2])
                shoulder_width = np.linalg.norm(sh_vec)
                # Estimate a local subject scale from available keypoints (bounding-box diagonal)
                pts_for_scale = []
                for p in (left_sh, right_sh, left_hip, right_hip, nose):
                    if p is None:
                        continue
                    pa = np.asarray(p)
                    if pa.shape[0] >= 3 and abs(pa[0]) < 1e-6:
                        pts_for_scale.append(pa[1:3])
                    else:
                        pts_for_scale.append(pa[:2])
                if pts_for_scale:
                    P = np.stack(pts_for_scale, axis=0)
                    bb_min = P.min(axis=0)
                    bb_max = P.max(axis=0)
                    subject_scale = float(np.linalg.norm(bb_max - bb_min))
                else:
                    subject_scale = float(shoulder_width)

                # Fallback guard
                if subject_scale < 1e-6:
                    subject_scale = float(max(shoulder_width, 1e-6))

                # Expected fraction of subject-scale occupied by shoulders (tunable)
                expected_shoulder_frac = 0.25
                # Normalize shoulder width by expected fraction of subject scale to get a scale-invariant confidence
                conf = float(np.clip(shoulder_width / (expected_shoulder_frac * subject_scale + 1e-8), 0.0, 1.0))
                method = 'shoulder-nose'

        else:
            pts = []
            for p in (left_sh, right_sh, left_hip, right_hip):
                if p is not None:
                    pa = np.asarray(p)
                    # prefer axes 1:3 when a leading constant axis is present
                    if pa.shape[0] >= 3 and abs(pa[0]) < 1e-6:
                        pts.append(pa[1:3])
                    else:
                        pts.append(pa[:2])
            if len(pts) >= 2:
                P = np.stack(pts, axis=0)
                Pc = P - P.mean(axis=0, keepdims=True)
                C = Pc.T @ Pc
                w, v = np.linalg.eigh(C)
                pc = v[:, np.argmax(w)]
                # pc is in [depth, lateral] space (x, y); compute yaw = atan2(lateral, depth)
                yaw = math.degrees(math.atan2(float(pc[1]), float(pc[0])))
                conf = min(1.0, len(pts) / 4.0)
                method = 'pca'

        yaws.append(yaw)
        confs.append(float(conf))
        methods.append(method)

    yaws = np.array(yaws, dtype=np.float32)
    confs = np.array(confs, dtype=np.float32)
    facs = []

    if smooth and len(yaws) > 0:
        s = yaws.copy()
        for j in range(1, len(s)):
            a_prev = math.radians(s[j-1])
            a_curr = math.radians(s[j])
            diff = ((a_curr - a_prev + math.pi) % (2*math.pi)) - math.pi
            a_sm = a_prev + alpha * diff
            s[j] = math.degrees(a_sm)
        yaws = s

    # Compute facing orientation using facial-keypoint presence as a heuristic
    def _get_by_name(k, frame_idx):
        orig = key_map.get(k)
        return _get_point_at_frame(d.get(orig) if orig is not None else None, frame_idx)

    for j, frame_idx in enumerate(idxs):
        facing_flag = 0
        # Check facial keypoint presence (nose/eyes/ears)
        facial_pts = ['nose', 'eye_left', 'eye_right', 'ear_left', 'ear_right']
        facial_count = 0
        for fk in facial_pts:
            p = _get_by_name(fk, frame_idx)
            if p is not None:
                # count if coordinates are finite
                if np.isfinite(p).all():
                    facial_count += 1

        if facial_count >= 1:
            # at least one facial keypoint visible -> likely facing camera
            facing_flag = 1
        else:
            # no facial points but shoulders exist -> likely facing away
            lsh = _get_by_name('shoulder_left', frame_idx)
            rsh = _get_by_name('shoulder_right', frame_idx)
            if lsh is not None and rsh is not None:
                facing_flag = -1

        facs.append(facing_flag)

    facs = np.array(facs, dtype=np.int8)

    return {'yaw_deg': yaws, 'confidence': confs, 'method': methods, 'frames': idxs, 'facing': facs}


def estimate_on_sample_frames(dptset, n=5, **kwargs):
    """Convenience: estimate on first `n` frames and return results."""
    return estimate_facing_from_dptset(dptset, frames=list(range(n)), **kwargs)
