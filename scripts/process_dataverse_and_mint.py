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

MODALITIES = ("pose", "gaze", "emotion", "traj", "robot")
POSE_DIM = len(KEYPOINT_ORDER) * 3
GAZE_DIM = 5
FEATURE_EMOTION_DIM = 7
EMOTION_ORDER = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "unknown"]
EMOTION_DIM = FEATURE_EMOTION_DIM + len(EMOTION_ORDER)
TRAJ_DIM = 6
ROBOT_DIM = 6


def _empty_series(length: int) -> np.ndarray:
    return np.full(length, np.nan, dtype=np.float32)


def normalize_pose(
    points: np.ndarray,
    valid_mask: np.ndarray,
    confidence: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool, np.ndarray, float]:
    if not valid_mask.any():
        feat = np.zeros(POSE_DIM, dtype=np.float32)
        return feat, False, np.zeros(2, dtype=np.float32), 1.0

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


def encode_gaze_dataverse(row: pd.Series, root: np.ndarray, scale: float) -> Tuple[np.ndarray, bool]:
    vector = np.zeros(GAZE_DIM, dtype=np.float32)
    required = {"gaze_pos_x", "gaze_pos_y"}
    if not required.issubset(row.index):
        return vector, False
    gx = row.get("gaze_pos_x", np.nan)
    gy = row.get("gaze_pos_y", np.nan)
    gz = row.get("gaze_pos_z", 0.0)
    if not (np.isfinite(gx) and np.isfinite(gy)):
        return vector, False
    vector[0] = (float(gx) - float(root[0])) / max(scale, 1e-3)
    vector[1] = (float(gy) - float(root[1])) / max(scale, 1e-3)
    vector[2] = float(gz) / max(scale, 1e-3) if np.isfinite(gz) else 0.0
    looking = row.get("looking_at_robot", 0)
    vector[4] = 1.0 if bool(looking) else 0.0
    return vector, True


def encode_robot_dataverse(row: pd.Series, root: np.ndarray, scale: float) -> Tuple[np.ndarray, bool]:
    vector = np.zeros(ROBOT_DIM, dtype=np.float32)
    base_x = row.get("robot_base_x", np.nan)
    base_y = row.get("robot_base_y", np.nan)
    base_yaw = row.get("robot_base_yaw", np.nan)
    head_x = row.get("robot_head_x", np.nan)
    head_y = row.get("robot_head_y", np.nan)
    head_yaw = row.get("robot_head_yaw", np.nan)
    if not (np.isfinite(base_x) and np.isfinite(base_y)):
        return vector, False
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


def compute_frame_labels(dataverse_row: pd.Series) -> int:
    for col in ("will_interact", "future_interaction", "future_interaction_4_sec", "interacting"):
        if col in dataverse_row.index and bool(dataverse_row.get(col, False)):
            return 1
    return 0


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
    pose_flat, pose_valid, root, scale = normalize_pose(points, valid_mask, confidence)

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
    if df_track.empty:
        return None
    df_local = _sort_dataverse_rows(df_track.copy())
    n = len(df_local)
    pose = np.zeros((n, POSE_DIM), dtype=np.float32)
    gaze = np.zeros((n, GAZE_DIM), dtype=np.float32)
    emotion = np.zeros((n, EMOTION_DIM), dtype=np.float32)
    traj = np.zeros((n, TRAJ_DIM), dtype=np.float32)
    robot = np.zeros((n, ROBOT_DIM), dtype=np.float32)
    modality_mask = np.zeros((n, len(MODALITIES)), dtype=np.float32)
    frame_labels = np.zeros(n, dtype=np.int8)

    keypoint_data: Dict[Tuple[str, str], np.ndarray] = {}
    for name, alias in DATAVERSE_JOINT_MAP.items():
        col_x = f"{alias}_x"
        col_y = f"{alias}_y"
        keypoint_data[(name, "x")] = (
            df_local[col_x].to_numpy(dtype=np.float32) if col_x in df_local.columns else _empty_series(n)
        )
        keypoint_data[(name, "y")] = (
            df_local[col_y].to_numpy(dtype=np.float32) if col_y in df_local.columns else _empty_series(n)
        )

    prev_root: Optional[np.ndarray] = None
    prev_scale = 1.0

    for idx in range(n):
        points = np.full((len(KEYPOINT_ORDER), 2), np.nan, dtype=np.float32)
        valid_mask = np.zeros(len(KEYPOINT_ORDER), dtype=bool)
        for name in KEYPOINT_ORDER:
            x = keypoint_data[(name, "x")][idx]
            y = keypoint_data[(name, "y")][idx]
            if np.isfinite(x) and np.isfinite(y):
                kp_idx = KEYPOINT_INDEX[name]
                points[kp_idx] = (x, y)
                valid_mask[kp_idx] = True
        pose_flat, pose_valid, root, scale = normalize_pose(points, valid_mask)
        pose[idx] = pose_flat
        modality_mask[idx, MODALITIES.index("pose")] = 1.0 if pose_valid else 0.0

        row = df_local.iloc[idx]
        gaze_vec, gaze_valid = encode_gaze_dataverse(row, root, scale)
        gaze[idx] = gaze_vec
        modality_mask[idx, MODALITIES.index("gaze")] = 1.0 if gaze_valid else 0.0

        emotion_vec, emotion_valid = encode_emotion_vector(None, None)
        emotion[idx] = emotion_vec
        modality_mask[idx, MODALITIES.index("emotion")] = 1.0 if emotion_valid else 0.0

        traj[idx] = compute_traj_features(root, prev_root, scale, prev_scale)
        modality_mask[idx, MODALITIES.index("traj")] = 1.0 if pose_valid else 0.0

        robot_vec, robot_valid = encode_robot_dataverse(row, root, scale)
        robot[idx] = robot_vec
        modality_mask[idx, MODALITIES.index("robot")] = 1.0 if robot_valid else 0.0

        frame_labels[idx] = compute_frame_labels(row)
        if pose_valid:
            prev_root = root
            prev_scale = scale

    intent_label = np.array(int(frame_labels.any()), dtype=np.int8)

    return {
        "pose": pose,
        "gaze": gaze,
        "emotion": emotion,
        "traj": traj,
        "robot": robot,
        "modality_mask": modality_mask,
        "frame_labels": frame_labels.astype(np.int8),
        "intent_label": intent_label,
        "frame_type": np.array(["robot_first_person"]),
        "sequence_id": np.array([_sanitize_sequence_id(sequence_id)]),
    }


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
        intent_label = np.array(int(labels.any()), dtype=np.int8)
        return {
            "pose": pose,
            "gaze": gaze,
            "emotion": emotion,
            "traj": traj,
            "robot": robot,
            "modality_mask": mask,
            "frame_labels": labels,
            "intent_label": intent_label,
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
                np.savez_compressed(
                    str(out_path),
                    pose=payload["pose"],
                    gaze=payload["gaze"],
                    emotion=payload["emotion"],
                    traj=payload["traj"],
                    robot=payload["robot"],
                    modality_mask=payload["modality_mask"],
                    frame_labels=payload["frame_labels"],
                    intent_label=payload["intent_label"],
                    frame_type=payload["frame_type"],
                    sequence_id=payload["sequence_id"],
                )
                stats["converted"] += 1
                stats["tracks_exported"] += 1
                traj_arr = payload["traj"].astype(np.float32)
                mask_arr = payload["modality_mask"].astype(np.float64)
                if traj_sum is None:
                    traj_sum = np.zeros(traj_arr.shape[1], dtype=float)
                    traj_sumsq = np.zeros(traj_arr.shape[1], dtype=float)
                traj_sum += traj_arr.sum(axis=0)
                traj_sumsq += (traj_arr ** 2).sum(axis=0)
                modality_sum += mask_arr.sum(axis=0)
                total_frames += traj_arr.shape[0]
        except Exception as exc:
            stats["failures"].append({"file": str(csv_path), "error": str(exc)})

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
                np.savez_compressed(
                    str(out_path),
                    pose=payload["pose"],
                    gaze=payload["gaze"],
                    emotion=payload["emotion"],
                    traj=payload["traj"],
                    robot=payload["robot"],
                    modality_mask=payload["modality_mask"],
                    frame_labels=payload["frame_labels"],
                    intent_label=payload["intent_label"],
                    frame_type=payload["frame_type"],
                    sequence_id=sequence_id,
                )
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
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

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
