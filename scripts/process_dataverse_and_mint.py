#!/usr/bin/env python3
"""Convert Dataverse CSVs and MINT-RVAE sessions into a unified multimodal NPZ format."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd

DEFAULT_IOU_THRESHOLD = 0.35
FUTURE_HORIZON_SECONDS = 5.0
DATAVERSE_FPS = 5.0
MINT_FPS = 5.0

KEYPOINT_ORDER = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
KEYPOINT_INDEX = {name: idx for idx, name in enumerate(KEYPOINT_ORDER)}

DATAVERSE_JOINT_MAP = {
    "nose": "nose",
    "left_eye": "eye_left",
    "right_eye": "eye_right",
    "left_ear": "ear_left",
    "right_ear": "ear_right",
    "left_shoulder": "shoulder_left",
    "right_shoulder": "shoulder_right",
    "left_elbow": "elbow_left",
    "right_elbow": "elbow_right",
    "left_wrist": "hand_left",
    "right_wrist": "hand_right",
    "left_hip": "hip_left",
    "right_hip": "hip_right",
    "left_knee": "knee_left",
    "right_knee": "knee_right",
    "left_ankle": "ankle_left",
    "right_ankle": "ankle_right",
}

DISABLE_DATAVERSE_GAZE = True  # Dataverse "gaze" refers to robot, not person
DISABLE_DATAVERSE_TRAJ = True  # Drop depth-derived trajectories to match monocular deployment
PROJECT_DATAVERSE_TO_2D = True  # Project 3D joints to image plane by dividing by depth
TRUNCATE_DATAVERSE_AT_FIRST_INTERACTION = True  # Keep frames only up to the first positive intent label
STORE_DATAVERSE_DEPTH = True  # Preserve raw 3D joints for analysis/visualization

KINECT_FX = 525.0
KINECT_FY = 525.0
KINECT_CX = 319.5
KINECT_CY = 239.5

MODALITIES = ("pose", "gaze", "emotion", "traj", "robot")

# Pose feature configuration. Default is 2D XY plus confidence (3 values per keypoint).
# When Dataverse 3D mode is enabled we switch to XYZ plus confidence (4 values per keypoint).
POSE_COMPONENTS = 2
POSE_DIM = len(KEYPOINT_ORDER) * (POSE_COMPONENTS + 1)
GAZE_DIM = 5
FEATURE_EMOTION_DIM = 7
EMOTION_ORDER = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "unknown"]
EMOTION_DIM = FEATURE_EMOTION_DIM + len(EMOTION_ORDER)
TRAJ_DIM = 6
ROBOT_DIM = 6


def _empty_series(length: int) -> np.ndarray:
    return np.full(length, np.nan, dtype=np.float32)


def _set_pose_components(num_components: int) -> None:
    global POSE_COMPONENTS, POSE_DIM
    POSE_COMPONENTS = max(2, int(num_components))
    POSE_DIM = len(KEYPOINT_ORDER) * (POSE_COMPONENTS + 1)


def normalize_pose(
    points: np.ndarray,
    valid_mask: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    expected_components: Optional[int] = None,
) -> Tuple[np.ndarray, bool, np.ndarray, float]:
    components = expected_components if expected_components is not None else POSE_COMPONENTS
    if points.shape[1] != components:
        raise ValueError(
            f"normalize_pose expected {components} components per joint but received {points.shape[1]}"
        )
    pose_dim = len(KEYPOINT_ORDER) * (components + 1)
    if not valid_mask.any():
        feat = np.zeros(pose_dim, dtype=np.float32)
        return feat, False, np.zeros(components, dtype=np.float32), 1.0

    hips = []
    for name in ("left_hip", "right_hip"):
        idx = KEYPOINT_INDEX[name]
        if valid_mask[idx]:
            hips.append(points[idx])
    if hips:
        root = np.mean(np.stack(hips, axis=0), axis=0)
    else:
        root = points[valid_mask][0]

    centered = np.zeros_like(points, dtype=np.float32)
    centered[valid_mask] = points[valid_mask] - root

    scale_candidates = []
    left_shoulder = KEYPOINT_INDEX["left_shoulder"]
    right_shoulder = KEYPOINT_INDEX["right_shoulder"]
    if valid_mask[left_shoulder] and valid_mask[right_shoulder]:
        scale_candidates.append(np.linalg.norm(points[left_shoulder] - points[right_shoulder]))
    left_hip = KEYPOINT_INDEX["left_hip"]
    right_hip = KEYPOINT_INDEX["right_hip"]
    if valid_mask[left_hip] and valid_mask[right_hip]:
        scale_candidates.append(np.linalg.norm(points[left_hip] - points[right_hip]))
    scale = float(np.median(scale_candidates)) if scale_candidates else 1.0
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 1.0

    normalized = centered / scale
    if confidence is not None:
        conf_vec = np.where(valid_mask, confidence, 0.0).astype(np.float32)
    else:
        conf_vec = valid_mask.astype(np.float32)
    feat = np.concatenate([normalized, conf_vec.reshape(-1, 1)], axis=1).reshape(-1).astype(np.float32)
    return feat, True, root.astype(np.float32), scale


def get_robot_frame_transform(row: pd.Series) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    base_x = row.get("robot_base_x", np.nan)
    base_y = row.get("robot_base_y", np.nan)
    if not (np.isfinite(base_x) and np.isfinite(base_y)):
        return None
    base_yaw = row.get("robot_base_yaw", 0.0)
    yaw = float(base_yaw) if np.isfinite(base_yaw) else 0.0
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    translation = np.array([float(base_x), float(base_y)], dtype=np.float32)
    rotation = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]], dtype=np.float32)
    return translation, rotation


def get_camera_frame_transform(row: pd.Series) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    head_x = row.get("robot_head_x", np.nan)
    head_y = row.get("robot_head_y", np.nan)
    if not (np.isfinite(head_x) and np.isfinite(head_y)):
        return None
    head_yaw = row.get("robot_head_yaw", row.get("robot_base_yaw", 0.0))
    yaw = float(head_yaw) if np.isfinite(head_yaw) else 0.0
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    translation = np.array([float(head_x), float(head_y)], dtype=np.float32)
    rotation = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]], dtype=np.float32)
    head_z = row.get("robot_head_z", 0.0)
    head_height = float(head_z) if np.isfinite(head_z) else 0.0
    return translation, rotation, head_height


def apply_robot_frame_transform(
    points: np.ndarray,
    valid_mask: np.ndarray,
    transform: Optional[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    if transform is None:
        return points
    translation, rotation = transform
    result = points.copy()
    valid = valid_mask & np.all(np.isfinite(points), axis=1)
    if not np.any(valid):
        return result
    relative = points[valid] - translation
    result[valid] = relative @ rotation.T
    return result


def transform_xy_to_robot(
    x: float,
    y: float,
    transform: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, float]:
    if transform is None or not (np.isfinite(x) and np.isfinite(y)):
        return float(x), float(y)
    translation, rotation = transform
    vec = np.array([float(x), float(y)], dtype=np.float32) - translation
    rotated = vec @ rotation.T
    return float(rotated[0]), float(rotated[1])


def perspective_project(
    points3d: np.ndarray,
    depth_axis: int = 1,
    lateral_axis: int = 0,
    vertical_axis: int = 2,
    min_depth: float = 1e-3,
    fx: float = KINECT_FX,
    fy: float = KINECT_FY,
    cx: float = KINECT_CX,
    cy: float = KINECT_CY,
) -> Tuple[np.ndarray, np.ndarray]:
    if points3d.ndim != 2 or points3d.shape[1] < 3:
        raise ValueError("points3d must have shape (N, 3)")
    depth = points3d[:, depth_axis]
    lateral = points3d[:, lateral_axis]
    vertical = points3d[:, vertical_axis]
    projected = np.full((points3d.shape[0], 2), np.nan, dtype=np.float32)
    valid = (
        np.isfinite(depth)
        & np.isfinite(lateral)
        & np.isfinite(vertical)
        & (depth > min_depth)
    )
    if np.any(valid):
        inv_depth = 1.0 / depth[valid]
        projected[valid, 0] = fx * lateral[valid] * inv_depth + cx
        projected[valid, 1] = cy - fy * vertical[valid] * inv_depth
    return projected, valid


def compute_traj_features(
    current_root: np.ndarray,
    prev_root: Optional[np.ndarray],
    current_scale: float,
    prev_scale: float,
) -> np.ndarray:
    if prev_root is None or not np.all(np.isfinite(prev_root)):
        delta = np.zeros(2, dtype=np.float32)
        speed = 0.0
        heading_cos = 1.0
        heading_sin = 0.0
    else:
        norm_scale = max((current_scale + prev_scale) * 0.5, 1e-3)
        delta = (current_root - prev_root) / norm_scale
        speed = float(np.linalg.norm(delta))
        heading = math.atan2(float(delta[1]), float(delta[0])) if speed > 1e-6 else 0.0
        heading_cos = math.cos(heading)
        heading_sin = math.sin(heading)
    return np.array([
        float(delta[0]),
        float(delta[1]),
        speed,
        heading_cos,
        heading_sin,
        float(current_scale),
    ], dtype=np.float32)


def encode_emotion_vector(label: Optional[str], features: Optional[Sequence[float]]) -> Tuple[np.ndarray, bool]:
    feat_vec = np.zeros(FEATURE_EMOTION_DIM, dtype=np.float32)
    if features is not None:
        arr = np.asarray(features, dtype=np.float32).reshape(-1)
        feat_vec[: min(arr.size, FEATURE_EMOTION_DIM)] = arr[:FEATURE_EMOTION_DIM]
        emotion_features_available = True
    else:
        emotion_features_available = False

    one_hot = np.zeros(len(EMOTION_ORDER), dtype=np.float32)
    if label is not None:
        label_lower = str(label).strip().lower()
        for idx, token in enumerate(EMOTION_ORDER):
            if token == label_lower:
                one_hot[idx] = 1.0
                break
    emotion = np.concatenate([feat_vec, one_hot], axis=0)
    mask = emotion_features_available or bool(one_hot.sum())
    return emotion, mask


def encode_gaze_dataverse(
    row: pd.Series,
    root: np.ndarray,
    scale: float,
    transform: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, bool]:
    vector = np.zeros(GAZE_DIM, dtype=np.float32)
    required = {"gaze_pos_x", "gaze_pos_y"}
    if not required.issubset(row.index):
        return vector, False
    gx = row.get("gaze_pos_x", np.nan)
    gy = row.get("gaze_pos_y", np.nan)
    gz = row.get("gaze_pos_z", 0.0)
    if not (np.isfinite(gx) and np.isfinite(gy)):
        return vector, False
    if transform is not None:
        gx, gy = transform_xy_to_robot(gx, gy, transform)
    vector[0] = (float(gx) - float(root[0])) / max(scale, 1e-3)
    vector[1] = (float(gy) - float(root[1])) / max(scale, 1e-3)
    vector[2] = float(gz) / max(scale, 1e-3) if np.isfinite(gz) else 0.0
    looking = row.get("looking_at_robot", 0)
    vector[4] = 1.0 if bool(looking) else 0.0
    return vector, True


def encode_robot_dataverse(
    row: pd.Series,
    root: np.ndarray,
    scale: float,
    transform: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, bool]:
    vector = np.zeros(ROBOT_DIM, dtype=np.float32)
    base_x = row.get("robot_base_x", np.nan)
    base_y = row.get("robot_base_y", np.nan)
    base_yaw = row.get("robot_base_yaw", np.nan)
    head_x = row.get("robot_head_x", np.nan)
    head_y = row.get("robot_head_y", np.nan)
    head_yaw = row.get("robot_head_yaw", np.nan)
    if not (np.isfinite(base_x) and np.isfinite(base_y)):
        return vector, False
    if transform is not None:
        base_x, base_y = transform_xy_to_robot(base_x, base_y, transform)
        if np.isfinite(head_x) and np.isfinite(head_y):
            head_x, head_y = transform_xy_to_robot(head_x, head_y, transform)
    norm = max(scale, 1e-3)
    base_dx = (float(base_x) - float(root[0])) / norm
    base_dy = (float(base_y) - float(root[1])) / norm
    base_dist = math.hypot(base_dx, base_dy)
    yaw = float(base_yaw) if np.isfinite(base_yaw) else 0.0
    head_dx = (float(head_x) - float(root[0])) / norm if np.isfinite(head_x) else 0.0
    head_dy = (float(head_y) - float(root[1])) / norm if np.isfinite(head_y) else 0.0
    head_yaw_val = float(head_yaw) if np.isfinite(head_yaw) else 0.0
    vector[:] = [base_dx, base_dy, base_dist, math.sin(yaw), math.cos(yaw), math.sin(head_yaw_val)]
    return vector, True


def extract_interaction_flag(dataverse_row: pd.Series) -> int:
    for col in ("interacting", "interaction", "interact"):
        if col in dataverse_row.index:
            return 1 if bool(dataverse_row.get(col, False)) else 0
    for col in ("will_interact", "future_interaction", "future_interaction_4_sec"):
        if col in dataverse_row.index:
            return 1 if bool(dataverse_row.get(col, False)) else 0
    return 0


def compute_future_targets(labels: np.ndarray, fps: float, horizon_seconds: float = FUTURE_HORIZON_SECONDS) -> np.ndarray:
    if labels.size == 0:
        return np.zeros(0, dtype=np.int8)
    horizon_frames = max(1, int(math.ceil(fps * horizon_seconds)))
    csum = np.concatenate(([0], labels.astype(np.int64).cumsum()))
    future = np.zeros_like(labels, dtype=np.int8)
    n = labels.shape[0]
    for idx in range(n):
        end = min(n, idx + horizon_frames)
        future[idx] = 1 if (csum[end] - csum[idx]) > 0 else 0
    return future


def first_positive_cut(frame_labels: np.ndarray) -> Optional[int]:
    pos = np.flatnonzero(frame_labels.astype(np.int8) > 0)
    if pos.size == 0:
        return None
    return int(pos[0] + 1)  # include the first interacting frame


def safe_box(data: object) -> np.ndarray:
    arr = np.zeros(4, dtype=np.float32)
    if data is None:
        return arr
    try:
        vals = np.asarray(data, dtype=np.float32).reshape(-1)
    except Exception:
        return arr
    if vals.size >= 4:
        arr[:] = vals[:4]
    return arr


def encode_mint_detection(
    det: Dict[str, object],
    prev_root: Optional[np.ndarray],
    prev_scale: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    int,
]:
    raw_pose = det.get("pose")
    if raw_pose is None:
        points = np.full((len(KEYPOINT_ORDER), 2), np.nan, dtype=np.float32)
        confidence = np.zeros(len(KEYPOINT_ORDER), dtype=np.float32)
    else:
        pose_arr = np.asarray(raw_pose, dtype=np.float32)
        points = np.full((len(KEYPOINT_ORDER), 2), np.nan, dtype=np.float32)
        confidence = np.zeros(len(KEYPOINT_ORDER), dtype=np.float32)
        for idx in range(min(pose_arr.shape[0], len(KEYPOINT_ORDER))):
            x, y = pose_arr[idx, 0], pose_arr[idx, 1]
            if np.isfinite(x) and np.isfinite(y):
                points[idx] = (x, y)
                confidence[idx] = pose_arr[idx, 2] if pose_arr.shape[1] >= 3 else 1.0
    valid_mask = confidence > 0.01
    pose_flat, pose_valid, root, scale = normalize_pose(points, valid_mask, confidence, expected_components=2)

    gaze_vec = np.zeros(GAZE_DIM, dtype=np.float32)
    gaze_valid = False
    for key in ("gaze_vector", "gaze_direction", "gaze", "gaze_pos", "gaze_point"):
        raw = det.get(key)
        if raw is None:
            continue
        try:
            arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if arr.size == 0:
            continue
        length = min(arr.size, 3)
        gaze_vec[:length] = arr[:length]
        gaze_valid = True
        break
    stability = str(det.get("gaze_stability", "None")).strip().lower()
    if stability == "low":
        gaze_vec[3] = 1.0
    elif stability == "high":
        gaze_vec[4] = 1.0

    feat_emotion_raw: Any = det.get("feature_emotion")
    if isinstance(feat_emotion_raw, (list, tuple, np.ndarray)):
        feat_emotion = cast(Sequence[float], feat_emotion_raw)
    else:
        feat_emotion = None
    emotion_label_raw = det.get("emotion")
    emotion_label = str(emotion_label_raw) if emotion_label_raw is not None else None
    emotion_vec, emotion_valid = encode_emotion_vector(emotion_label, feat_emotion)

    traj_vec = compute_traj_features(root, prev_root, scale, prev_scale)
    robot_vec = np.zeros(ROBOT_DIM, dtype=np.float32)

    mask_vec = np.array([
        1.0 if pose_valid else 0.0,
        1.0 if gaze_valid else 0.0,
        1.0 if emotion_valid else 0.0,
        1.0 if pose_valid else 0.0,
        0.0,
    ], dtype=np.float32)

    label = 1 if str(det.get("label", "no_interaction")).strip().lower() == "interaction" else 0

    return (
        pose_flat,
        gaze_vec,
        emotion_vec,
        traj_vec,
        robot_vec,
        mask_vec,
        root,
        scale,
        label,
    )


def _sort_dataverse_rows(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("resampled_timestamp", "timestamp", "shutter_timestamp", "tf_timestamp"):
        if col in df.columns:
            return df.sort_values(col, kind="mergesort").reset_index(drop=True)
    return df.reset_index(drop=True)


def _sanitize_sequence_id(value: str) -> str:
    safe = value.replace("/", "_").replace("\\", "_")
    return safe


def _format_pid(pid_value: object, fallback_idx: int) -> str:
    if pid_value is None:
        return f"track{fallback_idx:03d}"
    try:
        if isinstance(pid_value, (int, np.integer)):
            return f"{int(pid_value):06d}"
        if isinstance(pid_value, (float, np.floating)) and np.isfinite(pid_value):
            return f"{int(pid_value):06d}"
    except Exception:
        pass
    pid_str = str(pid_value).strip()
    if not pid_str or pid_str.lower() == "nan":
        return f"track{fallback_idx:03d}"
    return pid_str.replace(" ", "_")


def _build_dataverse_track(
    df_track: pd.DataFrame,
    csv_path: Path,
    sequence_id: str,
) -> Optional[Dict[str, np.ndarray]]:
    """Build track using robot-frame 3D XYZ for YOLO keypoints.
    
    Intent label logic:
    - will_interact == True → label as intent=1
    - will_interact == False → label as intent=0
    - Keep full sequences, filter out sequences < 10 frames (insufficient temporal context)
    """
    if df_track.empty:
        return None
    df_local = _sort_dataverse_rows(df_track.copy())
    
    # Determine sequence-level intent from will_interact column
    will_interact = False
    if 'will_interact' in df_local.columns:
        will_interact = bool(df_local['will_interact'].iloc[0])
    
    n = len(df_local)
    
    # Store 3D robot-frame poses (XYZ per keypoint, 17 keypoints = 51 values + 17 confidence = 68 total)
    pose_data = np.zeros((n, len(KEYPOINT_ORDER), 3), dtype=np.float32)
    pose_valid_mask = np.zeros((n, len(KEYPOINT_ORDER)), dtype=bool)
    
    # Extract keypoint data
    for idx in range(n):
        row = df_local.iloc[idx]
        robot_transform = get_robot_frame_transform(row)
        
        points_xy = np.full((len(KEYPOINT_ORDER), 2), np.nan, dtype=np.float32)
        points_z = np.full(len(KEYPOINT_ORDER), np.nan, dtype=np.float32)
        
        for name in KEYPOINT_ORDER:
            alias = DATAVERSE_JOINT_MAP.get(name)
            if alias is None:
                continue
            x = row.get(f'{alias}_x', np.nan)
            y = row.get(f'{alias}_y', np.nan)
            z = row.get(f'{alias}_z', np.nan)
            
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                kp_idx = KEYPOINT_INDEX[name]
                points_xy[kp_idx] = (x, y)
                points_z[kp_idx] = z
        
        # Transform to robot frame
        valid_2d = np.isfinite(points_xy).all(axis=1)
        points_xy_robot = apply_robot_frame_transform(points_xy, valid_2d, robot_transform)
        
        # Combine into 3D
        for kp_idx in range(len(KEYPOINT_ORDER)):
            if valid_2d[kp_idx] and np.isfinite(points_z[kp_idx]):
                pose_data[idx, kp_idx, 0] = points_xy_robot[kp_idx, 0]
                pose_data[idx, kp_idx, 1] = points_xy_robot[kp_idx, 1]
                pose_data[idx, kp_idx, 2] = points_z[kp_idx]
                pose_valid_mask[idx, kp_idx] = True
    
    # Skip sequences that are too short (need minimum temporal context)
    MIN_SEQUENCE_LENGTH = 10
    if n < MIN_SEQUENCE_LENGTH:
        return None
    
    # Keep full sequences - no truncation
    # Label is based on will_interact flag
    
    # Flatten pose data: for each frame, concatenate [x,y,z,conf] for all 17 keypoints
    T = pose_data.shape[0]
    pose_flat = np.zeros((T, 68), dtype=np.float32)  # 17 keypoints * 4 values (x,y,z,conf)
    
    for t in range(T):
        for kp in range(len(KEYPOINT_ORDER)):
            offset = kp * 4
            if pose_valid_mask[t, kp]:
                pose_flat[t, offset:offset+3] = pose_data[t, kp]
                pose_flat[t, offset+3] = 1.0  # confidence = 1 if joint present
            # else leave as zeros (missing joint)
    
    # Create modality mask (only pose is available for Dataverse)
    modality_mask = np.zeros((T, len(MODALITIES)), dtype=np.float32)
    modality_mask[:, MODALITIES.index("pose")] = 1.0
    
    # Intent label
    intent_label = np.array(1 if will_interact else 0, dtype=np.int8)
    
    # Frame labels: all frames before interaction are 0, interaction frame is 1
    frame_labels = np.zeros(T, dtype=np.int8)
    if will_interact and T > 0:
        frame_labels[-1] = 1  # Last frame is the interaction frame
    
    payload = {
        "pose": pose_flat,
        "gaze": np.zeros((T, GAZE_DIM), dtype=np.float32),
        "emotion": np.zeros((T, EMOTION_DIM), dtype=np.float32),
        "traj": np.zeros((T, TRAJ_DIM), dtype=np.float32),
        "robot": np.zeros((T, ROBOT_DIM), dtype=np.float32),
        "modality_mask": modality_mask,
        "frame_labels": frame_labels,
        "future_labels": np.zeros(T, dtype=np.int8),  # Not used for now
        "intent_label": intent_label,
        "pose_root": np.zeros((T, 3), dtype=np.float32),  # Not used but kept for compatibility
        "pose_scale": np.ones(T, dtype=np.float32),  # Not used but kept for compatibility
        "frame_type": np.array(["robot_first_person"]),
        "sequence_id": np.array([_sanitize_sequence_id(sequence_id)]),
    }
    
    return payload


def convert_dataverse_csv(csv_path: Path) -> List[Dict[str, np.ndarray]]:
    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        return []

    if "pid" not in df.columns:
        payload = _build_dataverse_track(df, csv_path, csv_path.stem)
        return [payload] if payload is not None else []

    payloads: List[Dict[str, np.ndarray]] = []
    for fallback_idx, (pid, group) in enumerate(df.groupby("pid"), start=0):
        pid_str = _format_pid(pid, fallback_idx)
        sequence_id = f"{csv_path.stem}_pid{pid_str}"
        payload = _build_dataverse_track(group, csv_path, sequence_id)
        if payload is not None:
            payloads.append(payload)
    return payloads


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


@dataclass
class Track:
    track_id: int
    last_box: np.ndarray
    last_frame: int
    frames: List[int] = field(default_factory=list)
    pose_feats: List[np.ndarray] = field(default_factory=list)
    gaze_feats: List[np.ndarray] = field(default_factory=list)
    emotion_feats: List[np.ndarray] = field(default_factory=list)
    traj_feats: List[np.ndarray] = field(default_factory=list)
    robot_feats: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    last_root: Optional[np.ndarray] = None
    last_scale: float = 1.0
    missed: int = 0
    root_positions: List[np.ndarray] = field(default_factory=list)
    pose_scales: List[float] = field(default_factory=list)

    def add(self, frame_idx: int, det: Dict[str, object]) -> None:
        (
            pose_feature,
            gaze_feature,
            emotion_feature,
            traj_feature,
            robot_feature,
            mask_vec,
            root,
            scale,
            label,
        ) = encode_mint_detection(det, self.last_root, self.last_scale)
        self.frames.append(frame_idx)
        self.pose_feats.append(pose_feature)
        self.gaze_feats.append(gaze_feature)
        self.emotion_feats.append(emotion_feature)
        self.traj_feats.append(traj_feature)
        self.robot_feats.append(robot_feature)
        self.masks.append(mask_vec)
        self.labels.append(label)
        self.last_box = safe_box(det.get("box"))
        self.last_frame = frame_idx
        self.last_root = root
        self.last_scale = scale
        self.missed = 0
        self.root_positions.append(root.astype(np.float32))
        self.pose_scales.append(float(scale))

    def length(self) -> int:
        return len(self.frames)

    def to_npz_payload(self) -> Dict[str, np.ndarray]:
        pose = np.stack(self.pose_feats).astype(np.float32)
        gaze = np.stack(self.gaze_feats).astype(np.float32)
        emotion = np.stack(self.emotion_feats).astype(np.float32)
        traj = np.stack(self.traj_feats).astype(np.float32)
        robot = np.stack(self.robot_feats).astype(np.float32)
        mask = np.stack(self.masks).astype(np.float32)
        labels = np.asarray(self.labels, dtype=np.int8)
        roots = np.stack(self.root_positions).astype(np.float32)
        scales = np.asarray(self.pose_scales, dtype=np.float32)
        cut = first_positive_cut(labels)
        if cut is not None:
            pose = pose[:cut]
            gaze = gaze[:cut]
            emotion = emotion[:cut]
            traj = traj[:cut]
            robot = robot[:cut]
            mask = mask[:cut]
            labels = labels[:cut]
            roots = roots[:cut]
            scales = scales[:cut]
        future_labels = compute_future_targets(labels, fps=MINT_FPS)
        intent_label = np.array(int(labels.any()), dtype=np.int8)
        return {
            "pose": pose,
            "gaze": gaze,
            "emotion": emotion,
            "traj": traj,
            "robot": robot,
            "modality_mask": mask,
            "frame_labels": labels,
            "future_labels": future_labels,
            "intent_label": intent_label,
            "pose_root": roots,
            "pose_scale": scales,
            "frame_type": np.array(["third_person"]),
        }


def assign_tracks(
    frames: Sequence[object],
    iou_threshold: float,
    max_missed: int,
    min_track_len: int,
) -> List[Dict[str, np.ndarray]]:
    active: List[Track] = []
    finished: List[Track] = []
    next_track_id = 0

    for frame_idx, raw in enumerate(frames):
        detections = raw if isinstance(raw, list) else []
        for track in list(active):
            if frame_idx - track.last_frame > max_missed:
                finished.append(track)
                active.remove(track)

        candidates: List[Tuple[float, int, int]] = []
        for t_idx, track in enumerate(active):
            for d_idx, det in enumerate(detections):
                box = safe_box(det.get("box"))
                score = iou(track.last_box, box)
                if score >= iou_threshold:
                    candidates.append((score, t_idx, d_idx))
        candidates.sort(reverse=True, key=lambda item: item[0])

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for score, t_idx, d_idx in candidates:
            if t_idx in used_tracks or d_idx in used_dets:
                continue
            track = active[t_idx]
            track.add(frame_idx, detections[d_idx])
            used_tracks.add(t_idx)
            used_dets.add(d_idx)

        survivors: List[Track] = []
        for idx, track in enumerate(active):
            if idx in used_tracks:
                survivors.append(track)
            else:
                track.missed += 1
                if track.missed > max_missed:
                    finished.append(track)
                else:
                    survivors.append(track)
        active = survivors

        for d_idx, det in enumerate(detections):
            if d_idx in used_dets:
                continue
            box = safe_box(det.get("box"))
            track = Track(track_id=next_track_id, last_box=box, last_frame=frame_idx)
            track.add(frame_idx, det)
            active.append(track)
            next_track_id += 1

    finished.extend(active)

    payloads = []
    for track in finished:
        if track.length() < min_track_len:
            continue
        try:
            payloads.append(track.to_npz_payload())
        except ValueError:
            continue
    return payloads


def process_dataverse(root: Path, out_dir: Path) -> Dict[str, object]:
    csv_files = sorted(root.rglob("*.csv"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "input_files": len(csv_files),
        "converted": 0,
        "failures": [],
        "output_dir": str(out_dir),
        "tracks_exported": 0,
    }

    traj_sum = None
    traj_sumsq = None
    modality_sum = np.zeros(len(MODALITIES), dtype=float)
    total_frames = 0

    for csv_path in csv_files:
        try:
            payloads = convert_dataverse_csv(csv_path)
            if not payloads:
                continue
            for payload in payloads:
                seq_tokens = payload.get("sequence_id")
                seq_name = str(seq_tokens[0]) if isinstance(seq_tokens, np.ndarray) else csv_path.stem
                out_path = out_dir / f"{seq_name}.npz"
                save_payload = {
                    "pose": payload["pose"],
                    "gaze": payload["gaze"],
                    "emotion": payload["emotion"],
                    "traj": payload["traj"],
                    "robot": payload["robot"],
                    "modality_mask": payload["modality_mask"],
                    "frame_labels": payload["frame_labels"],
                    "future_labels": payload["future_labels"],
                    "intent_label": payload["intent_label"],
                    "pose_root": payload["pose_root"],
                    "pose_scale": payload["pose_scale"],
                    "frame_type": payload["frame_type"],
                    "sequence_id": payload["sequence_id"],
                }
                if "pose_depth" in payload:
                    save_payload["pose_depth"] = payload["pose_depth"]
                np.savez_compressed(str(out_path), **save_payload)
                stats["converted"] += 1
                stats["tracks_exported"] += 1
                traj_arr = payload["traj"].astype(np.float32)
                mask_arr = payload["modality_mask"].astype(np.float64)
                if not DISABLE_DATAVERSE_TRAJ:
                    if traj_sum is None:
                        traj_sum = np.zeros(traj_arr.shape[1], dtype=float)
                        traj_sumsq = np.zeros(traj_arr.shape[1], dtype=float)
                    traj_sum += traj_arr.sum(axis=0)
                    traj_sumsq += (traj_arr ** 2).sum(axis=0)
                modality_sum += mask_arr.sum(axis=0)
                total_frames += traj_arr.shape[0]
        except Exception as exc:
            stats["failures"].append({"file": str(csv_path), "error": str(exc)})

    if total_frames > 0:
        stats["modality_coverage"] = {
            modality: float(modality_sum[idx] / total_frames)
            for idx, modality in enumerate(MODALITIES)
        }

    if (not DISABLE_DATAVERSE_TRAJ) and total_frames > 0 and traj_sum is not None and traj_sumsq is not None:
        mean = traj_sum / total_frames
        var = np.maximum(traj_sumsq / total_frames - mean ** 2, 0.0)
        stats["traj_mean"] = mean.tolist()
        stats["traj_std"] = np.sqrt(var).tolist()

    if total_frames > 0 and stats.get("modality_coverage"):
        metadata = {
            "modality_coverage": stats["modality_coverage"],
            "total_frames": total_frames,
        }
        if "traj_mean" in stats and "traj_std" in stats:
            metadata["traj_mean"] = stats["traj_mean"]
            metadata["traj_std"] = stats["traj_std"]
        with (out_dir / "metadata.json").open("w") as fh:
            json.dump(metadata, fh, indent=2)
    return stats


def process_mint(
    root: Path,
    out_dir: Path,
    min_track_len: int,
    max_missed: int,
    iou_threshold: float,
) -> Dict[str, object]:
    npz_files = sorted(root.rglob("feature_session_*.npz"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "input_files": len(npz_files),
        "tracks_exported": 0,
        "failures": [],
        "output_dir": str(out_dir),
    }
    traj_sum = None
    traj_sumsq = None
    modality_sum = np.zeros(len(MODALITIES), dtype=float)
    total_frames = 0

    for session_path in npz_files:
        try:
            arr = np.load(session_path, allow_pickle=True)
            frames = list(arr.get("dataset", []))
            payloads = assign_tracks(
                frames=frames,
                iou_threshold=iou_threshold,
                max_missed=max_missed,
                min_track_len=min_track_len,
            )
            for track_idx, payload in enumerate(payloads):
                sequence_id = np.array([f"{session_path.stem}_track{track_idx:03d}"])
                out_path = out_dir / f"{sequence_id[0]}.npz"
                save_payload = {
                    "pose": payload["pose"],
                    "gaze": payload["gaze"],
                    "emotion": payload["emotion"],
                    "traj": payload["traj"],
                    "robot": payload["robot"],
                    "modality_mask": payload["modality_mask"],
                    "frame_labels": payload["frame_labels"],
                    "future_labels": payload["future_labels"],
                    "intent_label": payload["intent_label"],
                    "pose_root": payload["pose_root"],
                    "pose_scale": payload["pose_scale"],
                    "frame_type": payload["frame_type"],
                    "sequence_id": sequence_id,
                }
                if "pose_depth" in payload:
                    save_payload["pose_depth"] = payload["pose_depth"]
                np.savez_compressed(str(out_path), **save_payload)
                traj_arr = payload["traj"].astype(np.float32)
                mask_arr = payload["modality_mask"].astype(np.float64)
                stats["tracks_exported"] += 1
                if traj_sum is None:
                    traj_sum = np.zeros(traj_arr.shape[1], dtype=float)
                    traj_sumsq = np.zeros(traj_arr.shape[1], dtype=float)
                traj_sum += traj_arr.sum(axis=0)
                traj_sumsq += (traj_arr ** 2).sum(axis=0)
                modality_sum += mask_arr.sum(axis=0)
                total_frames += traj_arr.shape[0]
        except Exception as exc:
            stats["failures"].append({"file": str(session_path), "error": str(exc)})

    if total_frames > 0 and traj_sum is not None and traj_sumsq is not None:
        mean = traj_sum / total_frames
        var = np.maximum(traj_sumsq / total_frames - mean ** 2, 0.0)
        stats["traj_mean"] = mean.tolist()
        stats["traj_std"] = np.sqrt(var).tolist()
        stats["modality_coverage"] = {
            modality: float(modality_sum[idx] / total_frames)
            for idx, modality in enumerate(MODALITIES)
        }
        with (out_dir / "metadata.json").open("w") as fh:
            json.dump(
                {
                    "traj_mean": stats["traj_mean"],
                    "traj_std": stats["traj_std"],
                    "modality_coverage": stats["modality_coverage"],
                    "total_frames": total_frames,
                },
                fh,
                indent=2,
            )
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process Dataverse and MINT-RVAE datasets into unified NPZ shards")
    parser.add_argument("--dataverse-root", type=str, default="datasets/dataverse_files", help="Root directory containing processed Dataverse CSV files")
    parser.add_argument(
        "--mint-root",
        type=str,
        default="datasets/MINT-RVAE-Dataset-for-multimodal-intent-prediction-in-human-robot-interaction-main/data",
        help="Root directory containing MINT-RVAE feature_session_*.npz files",
    )
    parser.add_argument("--output-dir", type=str, default="datasets/processed", help="Directory where processed NPZ files will be stored")
    parser.add_argument("--mint-min-track", type=int, default=20, help="Minimum number of frames per MINT track before export")
    parser.add_argument("--mint-max-missed", type=int, default=2, help="Allowed consecutive missed frames before a track is closed")
    parser.add_argument("--mint-iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD, help="IoU threshold for greedy track association")
    parser.add_argument(
        "--dataverse-pose-mode",
        choices=["2d", "3d"],
        default="2d",
        help="Select whether to project Dataverse joints into 2D pixel coordinates or keep robot-frame XYZ coordinates",
    )
    parser.add_argument(
        "--dataverse-keep-full",
        action="store_true",
        help="Keep entire Dataverse sequences instead of truncating at the first positive interaction frame",
    )
    return parser


def main() -> None:
    global PROJECT_DATAVERSE_TO_2D, TRUNCATE_DATAVERSE_AT_FIRST_INTERACTION
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.dataverse_pose_mode == "3d":
        PROJECT_DATAVERSE_TO_2D = False
        _set_pose_components(3)
    else:
        PROJECT_DATAVERSE_TO_2D = True
        _set_pose_components(2)
    TRUNCATE_DATAVERSE_AT_FIRST_INTERACTION = not args.dataverse_keep_full

    dataverse_root = Path(args.dataverse_root)
    mint_root = Path(args.mint_root)
    out_dir = Path(args.output_dir)
    dataverse_out = out_dir / "dataverse_npz"
    mint_out = out_dir / "mint_npz"

    dataverse_stats = process_dataverse(dataverse_root, dataverse_out)
    print(json.dumps({"dataverse": dataverse_stats}, indent=2))

    mint_stats = process_mint(
        mint_root,
        mint_out,
        min_track_len=args.mint_min_track,
        max_missed=args.mint_max_missed,
        iou_threshold=args.mint_iou_threshold,
    )
    print(json.dumps({"mint": mint_stats}, indent=2))

    summary = {
        "dataverse": dataverse_stats,
        "mint": mint_stats,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "processing_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote summary to {out_dir / 'processing_summary.json'}")


if __name__ == "__main__":
    main()
