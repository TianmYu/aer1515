import bisect
import os
import random
import re
from collections import OrderedDict
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

GAZE_DIM = 5
EMOTION_DIM = 15
TRAJ_DIM = 6
ROBOT_DIM = 6
MODALITY_NAMES = ("pose", "gaze", "emotion", "traj", "robot")


def _slice_with_pad(
    array: Optional[np.ndarray],
    start: int,
    length: int,
    feature_dim: Optional[int],
    dtype: Union[np.dtype, type],
) -> np.ndarray:
    """Slice a window and pad with zeros when the source is too short."""
    if array is None:
        if feature_dim is None:
            return np.zeros((length,), dtype=dtype)
        if feature_dim == 0:
            return np.zeros((length, 0), dtype=dtype)
        return np.zeros((length, feature_dim), dtype=dtype)

    arr = np.asarray(array)
    if feature_dim == 0:
        return np.zeros((length, 0), dtype=dtype)

    end = start + length
    actual = arr[start:end]
    copy_len = min(actual.shape[0], length)

    if arr.ndim == 1 or feature_dim is None:
        out = np.zeros((length,), dtype=dtype)
        if copy_len > 0:
            out[:copy_len] = actual[:copy_len].astype(dtype, copy=False)
        return out

    feat_dim = feature_dim if feature_dim is not None else arr.shape[1]
    out = np.zeros((length, feat_dim), dtype=dtype)
    if copy_len > 0 and actual.ndim > 1:
        width = min(feat_dim, actual.shape[1])
        out[:copy_len, :width] = actual[:copy_len, :width].astype(dtype, copy=False)
    return out


class MultimodalWindowDataset(Dataset):
    """Sliding-window loader for multimodal NPZ shards and legacy CSV files."""

    def __init__(
        self,
        data_paths: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
        seq_len: int,
        stride: int = 1,
        glob_pattern: Optional[str] = None,
        backend: str = "auto",
        shuffle: bool = False,
    max_files: Optional[int] = None,
    files: Optional[Sequence[Union[str, os.PathLike]]] = None,
    traj_norm_stats: Optional[Tuple[Iterable[float], Iterable[float]]] = None,
    ):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.shuffle = bool(shuffle)
        self.backend = backend
        self._df_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._df_cache_max = 8
        self.traj_mean = None
        self.traj_std = None
        if traj_norm_stats is not None:
            mean, std = traj_norm_stats
            self.traj_mean = np.asarray(mean, dtype=np.float32)
            self.traj_std = np.asarray(std, dtype=np.float32)

        if files is not None:
            input_files = [os.fspath(p) for p in files]
        else:
            if isinstance(data_paths, (str, os.PathLike)):
                bases = [os.fspath(data_paths)]
            else:
                bases = [os.fspath(p) for p in data_paths]

            input_files: List[str] = []
            for base in bases:
                if os.path.isdir(base):
                    target_backend = backend
                    if target_backend == "auto":
                        target_backend = "npz"
                    pattern = glob_pattern
                    if pattern is None:
                        pattern = "**/*.npz" if target_backend == "npz" else "**/*.csv"
                    input_files.extend(sorted(glob(os.path.join(base, pattern), recursive=True)))
                elif os.path.isfile(base):
                    input_files.append(base)

        if not input_files:
            raise RuntimeError("No input files found for dataset")

        if self.backend == "auto":
            self.backend = "npz" if any(f.lower().endswith(".npz") for f in input_files) else "csv"

        input_files = [
            f
            for f in input_files
            if (self.backend == "npz" and f.lower().endswith(".npz"))
            or (self.backend == "csv" and f.lower().endswith(".csv"))
        ]
        if not input_files:
            raise RuntimeError("No files matched the selected backend")

        if self.shuffle:
            random.shuffle(input_files)
        if max_files is not None and files is None:
            input_files = input_files[: int(max_files)]

        self.files = input_files
        self.file_infos = []
        self.cum_counts = [0]
        self._inspect_files()

    def _inspect_files(self) -> None:
        self.file_infos.clear()
        self.cum_counts = [0]
        for path_idx, path in enumerate(self.files):
            try:
                if self.backend == "npz":
                    with np.load(path, allow_pickle=False) as arr:
                        if "pose" in arr:
                            n = int(arr["pose"].shape[0])
                        elif "traj" in arr:
                            n = int(arr["traj"].shape[0])
                        elif "frame_labels" in arr:
                            n = int(arr["frame_labels"].shape[0])
                        else:
                            continue

                        intent_val = 0
                        frame_pos = 0
                        seq_id = Path(path).stem if "sequence_id" not in arr else str(arr["sequence_id"].reshape(-1)[0])
                        if "intent_label" in arr:
                            intent_val = int(arr["intent_label"].reshape(-1)[0])
                        elif "frame_labels" in arr:
                            intent_val = int(np.asarray(arr["frame_labels"], dtype=np.int64).max())

                        if "frame_labels" in arr:
                            frame_vals = np.asarray(arr["frame_labels"], dtype=np.int64).reshape(-1)
                            frame_pos = int(frame_vals.sum())
                        else:
                            frame_pos = intent_val * n
                else:
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        n = max(0, sum(1 for _ in fh) - 1)
                    intent_val = 0
                    frame_pos = 0
                    seq_id = Path(path).stem
            except Exception:
                if self.backend == "npz":
                    continue
                try:
                    df_fallback = pd.read_csv(path, low_memory=False)
                    n = len(df_fallback)
                except Exception:
                    continue
                intent_val = 0
                frame_pos = 0
                seq_id = Path(path).stem

            if n < self.seq_len:
                n_windows = 1
            else:
                n_windows = ((n - self.seq_len) // self.stride) + 1
            self.file_infos.append(
                {
                    "file": path,
                    "index": path_idx,
                    "n_rows": n,
                    "n_windows": n_windows,
                    "intent_label": int(intent_val),
                    "frame_positive": int(frame_pos),
                    "frame_total": int(n),
                    "sequence_id": seq_id,
                }
            )
            self.cum_counts.append(self.cum_counts[-1] + n_windows)

        if not self.file_infos:
            raise RuntimeError("No windows found in dataset. Check inputs and seq_len")

    def _empty_sample(self) -> Dict[str, Union[np.ndarray, float, int]]:
        return {
            "pose": np.zeros((self.seq_len, 0), dtype=np.float32),
            "gaze": np.zeros((self.seq_len, GAZE_DIM), dtype=np.float32),
            "emotion": np.zeros((self.seq_len, EMOTION_DIM), dtype=np.float32),
            "traj": np.zeros((self.seq_len, TRAJ_DIM), dtype=np.float32),
            "robot": np.zeros((self.seq_len, ROBOT_DIM), dtype=np.float32),
            "modality_mask": np.zeros((self.seq_len, len(MODALITY_NAMES)), dtype=np.float32),
            "frame_labels": np.zeros((self.seq_len,), dtype=np.int64),
            "label": 0,
            "future_label": 0,
            "intent_label": 0,
            "has_pose": 0.0,
            "has_gaze": 0.0,
            "has_emotion": 0.0,
            "has_traj": 0.0,
            "has_robot": 0.0,
            "source_index": -1,
            "window_index": -1,
            "sequence_label": 0,
        }

    def _normalize_traj(self, traj: np.ndarray) -> np.ndarray:
        if (
            self.traj_mean is None
            or self.traj_std is None
            or traj.shape[1] != self.traj_mean.shape[0]
        ):
            return traj
        std_safe = self.traj_std.copy()
        std_safe[std_safe <= 0] = 1.0
        return (traj - self.traj_mean.reshape(1, -1)) / std_safe.reshape(1, -1)

    def _read_window_npz(self, file_path: str, start: int) -> Dict[str, Union[np.ndarray, float, int]]:
        try:
            with np.load(file_path, allow_pickle=False) as arr:
                keys = set(arr.files)
                if {"modality_mask", "intent_label", "frame_labels"} & keys:
                    return self._read_window_multimodal(arr, start)
                return self._read_window_legacy(arr, start)
        except Exception:
            return self._empty_sample()

    def _read_window_multimodal(
        self,
        arr: np.lib.npyio.NpzFile,
        start: int,
    ) -> Dict[str, Union[np.ndarray, float, int]]:
        pose_data = arr.get("pose")
        pose_dim = pose_data.shape[1] if pose_data is not None and pose_data.ndim > 1 else 0
        pose = _slice_with_pad(pose_data, start, self.seq_len, pose_dim, np.float32)

        gaze = _slice_with_pad(arr.get("gaze"), start, self.seq_len, GAZE_DIM, np.float32)
        emotion = _slice_with_pad(arr.get("emotion"), start, self.seq_len, EMOTION_DIM, np.float32)

        traj_data = arr.get("traj")
        traj_dim = traj_data.shape[1] if traj_data is not None and traj_data.ndim > 1 else TRAJ_DIM
        traj = _slice_with_pad(traj_data, start, self.seq_len, traj_dim, np.float32)
        traj = self._normalize_traj(traj)

        robot = _slice_with_pad(arr.get("robot"), start, self.seq_len, ROBOT_DIM, np.float32)
        mask = _slice_with_pad(arr.get("modality_mask"), start, self.seq_len, len(MODALITY_NAMES), np.float32)
        frame_labels = _slice_with_pad(
            arr.get("frame_labels"), start, self.seq_len, None, np.int64
        ).astype(np.int64, copy=False)

        label = int(frame_labels.max()) if frame_labels.size else 0
        intent = int(arr["intent_label"].item()) if "intent_label" in arr else label

        def _presence(idx: int, tensor: Optional[np.ndarray]) -> float:
            if mask.shape[1] > idx:
                return float(mask[:, idx].max())
            if tensor is None:
                return 0.0
            return float(np.any(tensor))

        has_pose = _presence(0, pose_data)
        has_gaze = _presence(1, arr.get("gaze"))
        has_emotion = _presence(2, arr.get("emotion"))
        has_traj = _presence(3, traj_data)
        has_robot = _presence(4, arr.get("robot"))

        return {
            "pose": pose.astype(np.float32, copy=False),
            "gaze": gaze.astype(np.float32, copy=False),
            "emotion": emotion.astype(np.float32, copy=False),
            "traj": traj.astype(np.float32, copy=False),
            "robot": robot.astype(np.float32, copy=False),
            "modality_mask": mask.astype(np.float32, copy=False),
            "frame_labels": frame_labels,
            "label": label,
            "future_label": intent,
            "intent_label": intent,
            "has_pose": has_pose,
            "has_gaze": has_gaze,
            "has_emotion": has_emotion,
            "has_traj": has_traj,
            "has_robot": has_robot,
        }

    def _read_window_legacy(
        self,
        arr: np.lib.npyio.NpzFile,
        start: int,
    ) -> Dict[str, Union[np.ndarray, float, int]]:
        pose_data = arr.get("pose")
        pose_dim = pose_data.shape[1] if pose_data is not None and pose_data.ndim > 1 else 0
        pose = _slice_with_pad(pose_data, start, self.seq_len, pose_dim, np.float32)

        traj_data = arr.get("traj")
        traj_dim = traj_data.shape[1] if traj_data is not None and traj_data.ndim > 1 else TRAJ_DIM
        traj = _slice_with_pad(traj_data, start, self.seq_len, traj_dim, np.float32)
        traj = self._normalize_traj(traj)

        labels = _slice_with_pad(arr.get("label"), start, self.seq_len, None, np.int64)
        future = _slice_with_pad(arr.get("future_label"), start, self.seq_len, None, np.int64)
        label = int(labels.max()) if labels.size else 0
        future_label = int(future.max()) if future.size else label

        mask = np.zeros((self.seq_len, len(MODALITY_NAMES)), dtype=np.float32)
        if pose_dim > 0:
            mask[:, 0] = 1.0
        if traj.shape[1] > 0:
            mask[:, 3] = 1.0

        frame_labels = (
            labels.astype(np.int64, copy=False)
            if labels.size
            else np.zeros((self.seq_len,), dtype=np.int64)
        )

        return {
            "pose": pose.astype(np.float32, copy=False),
            "gaze": np.zeros((self.seq_len, GAZE_DIM), dtype=np.float32),
            "emotion": np.zeros((self.seq_len, EMOTION_DIM), dtype=np.float32),
            "traj": traj.astype(np.float32, copy=False),
            "robot": np.zeros((self.seq_len, ROBOT_DIM), dtype=np.float32),
            "modality_mask": mask,
            "frame_labels": frame_labels,
            "label": label,
            "future_label": future_label,
            "intent_label": future_label,
            "has_pose": 1.0 if pose_dim > 0 else 0.0,
            "has_gaze": 0.0,
            "has_emotion": 0.0,
            "has_traj": 1.0 if traj.shape[1] > 0 else 0.0,
            "has_robot": 0.0,
        }

    def _read_window_csv(self, file_path: str, start: int) -> Dict[str, Union[np.ndarray, float, int]]:
        df = self._df_cache.get(file_path)
        if df is None:
            try:
                df = pd.read_csv(file_path, low_memory=False)
            except Exception:
                df = pd.DataFrame()
            self._df_cache[file_path] = df
            if len(self._df_cache) > self._df_cache_max:
                self._df_cache.popitem(last=False)

        if df.shape[0] == 0:
            win = pd.DataFrame()
        else:
            win = df.iloc[start:start + self.seq_len].reset_index(drop=True)

        cols = sorted(c for c in win.columns if re.search(r"(_x|_y)$", c))
        pose = (
            win[cols].fillna(0.0).to_numpy(dtype=np.float32)
            if cols
            else np.zeros((self.seq_len, 0), dtype=np.float32)
        )

        if {"pelvis_x", "pelvis_y"} <= set(win.columns):
            traj = win[["pelvis_x", "pelvis_y"]].fillna(0.0).to_numpy(dtype=np.float32)
        elif {"cart_x", "cart_y"} <= set(win.columns):
            traj = win[["cart_x", "cart_y"]].fillna(0.0).to_numpy(dtype=np.float32)
        else:
            traj = np.zeros((self.seq_len, 2), dtype=np.float32)
        traj = self._normalize_traj(traj)

        if "future_interaction" in win.columns:
            futs = win["future_interaction"].fillna(False).astype(bool).to_numpy()
            future_label = int(bool(futs.any()))
        elif "will_interact" in win.columns:
            futs = win["will_interact"].fillna(False).astype(bool).to_numpy()
            future_label = int(bool(futs.any()))
        else:
            future_label = 0

        if future_label:
            label = 1
        elif "interacting" in win.columns:
            lbls = win["interacting"].fillna(False).astype(bool).to_numpy()
            label = int(bool(lbls.any()))
        else:
            label = 0

        frame_labels = np.full((self.seq_len,), label, dtype=np.int64)
        mask = np.zeros((self.seq_len, len(MODALITY_NAMES)), dtype=np.float32)
        if pose.shape[1] > 0:
            mask[:, 0] = 1.0
        if traj.shape[1] > 0:
            mask[:, 3] = 1.0

        return {
            "pose": pose,
            "gaze": np.zeros((self.seq_len, GAZE_DIM), dtype=np.float32),
            "emotion": np.zeros((self.seq_len, EMOTION_DIM), dtype=np.float32),
            "traj": traj,
            "robot": np.zeros((self.seq_len, ROBOT_DIM), dtype=np.float32),
            "modality_mask": mask,
            "frame_labels": frame_labels,
            "label": label,
            "future_label": future_label,
            "intent_label": future_label,
            "has_pose": 1.0 if pose.shape[1] > 0 else 0.0,
            "has_gaze": 0.0,
            "has_emotion": 0.0,
            "has_traj": 1.0 if traj.shape[1] > 0 else 0.0,
            "has_robot": 0.0,
        }

    def _read_window(self, file_path: str, start: int) -> Dict[str, Union[np.ndarray, float, int]]:
        if self.backend == "npz" and file_path.lower().endswith(".npz"):
            return self._read_window_npz(file_path, start)
        return self._read_window_csv(file_path, start)

    def __len__(self) -> int:
        return self.cum_counts[-1] if self.cum_counts else 0

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        file_idx = bisect.bisect_right(self.cum_counts, idx) - 1
        local_idx = idx - self.cum_counts[file_idx]
        start = local_idx * self.stride
        file_path = self.file_infos[file_idx]["file"]
        window = self._read_window(file_path, start)

        sample = {
            "pose": torch.from_numpy(window["pose"]).float(),
            "gaze": torch.from_numpy(window["gaze"]).float(),
            "emotion": torch.from_numpy(window["emotion"]).float(),
            "traj": torch.from_numpy(window["traj"]).float(),
            "robot": torch.from_numpy(window["robot"]).float(),
            "modality_mask": torch.from_numpy(window["modality_mask"]).float(),
            "frame_labels": torch.from_numpy(window["frame_labels"]).long(),
            "label": torch.tensor(int(window["label"]), dtype=torch.long),
            "future_label": torch.tensor(int(window["future_label"]), dtype=torch.long),
            "intent_label": torch.tensor(
                int(window.get("intent_label", window["future_label"])), dtype=torch.long
            ),
            "has_pose": torch.tensor(window["has_pose"], dtype=torch.float32),
            "has_gaze": torch.tensor(window["has_gaze"], dtype=torch.float32),
            "has_emotion": torch.tensor(window["has_emotion"], dtype=torch.float32),
            "has_traj": torch.tensor(window["has_traj"], dtype=torch.float32),
            "has_robot": torch.tensor(window["has_robot"], dtype=torch.float32),
            "sequence_label": torch.tensor(
                self.file_infos[file_idx].get("intent_label", int(window.get("intent_label", 0))),
                dtype=torch.long,
            ),
        }
        sample["source_index"] = torch.tensor(file_idx, dtype=torch.long)
        sample["window_index"] = torch.tensor(local_idx, dtype=torch.long)
        return sample

    def get_window_label_list(self, label_key: str = "intent_label") -> List[int]:
        labels: List[int] = []
        for info in self.file_infos:
            lbl = int(info.get(label_key, 0))
            labels.extend([lbl] * int(info.get("n_windows", 0)))
        return labels

    def get_label_counts(self, label_key: str = "intent_label", num_classes: int = 2) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.int64)
        for info in self.file_infos:
            lbl = int(info.get(label_key, 0))
            counts[lbl] += int(info.get("n_windows", 0))
        return counts


def collate_fn(batch: List[dict]):
    """Pad variable-width modalities and stack a batch."""

    def _pad_to_max(tensors: List[torch.Tensor]) -> torch.Tensor:
        max_dim = max(t.shape[1] for t in tensors)
        device = tensors[0].device
        dtype = tensors[0].dtype
        if max_dim == 0:
            return torch.zeros(len(tensors), tensors[0].shape[0], 0, dtype=dtype, device=device)
        T = tensors[0].shape[0]
        out = torch.zeros(len(tensors), T, max_dim, dtype=dtype, device=device)
        for i, tensor in enumerate(tensors):
            if tensor.shape[1] == 0:
                continue
            width = min(max_dim, tensor.shape[1])
            out[i, :, :width] = tensor[:, :width]
        return out

    pose_tensor = _pad_to_max([b["pose"] for b in batch])
    traj_tensor = _pad_to_max([b["traj"] for b in batch])
    gaze_tensor = torch.stack([b["gaze"] for b in batch]).float()
    emotion_tensor = torch.stack([b["emotion"] for b in batch]).float()
    robot_tensor = torch.stack([b["robot"] for b in batch]).float()
    modality_mask = torch.stack([b["modality_mask"] for b in batch]).float()
    frame_labels = torch.stack([b["frame_labels"] for b in batch]).long()

    has_pose = torch.stack([b["has_pose"] for b in batch]).float()
    has_gaze = torch.stack([b["has_gaze"] for b in batch]).float()
    has_emotion = torch.stack([b["has_emotion"] for b in batch]).float()
    has_traj = torch.stack([b["has_traj"] for b in batch]).float()
    has_robot = torch.stack([b["has_robot"] for b in batch]).float()

    labels = torch.stack([b["label"] for b in batch]).long()
    future_labels = torch.stack([b["future_label"] for b in batch]).long()
    intent_labels = torch.stack([b["intent_label"] for b in batch]).long()
    sequence_labels = torch.stack([b["sequence_label"] for b in batch]).long()
    source_indices = torch.stack([b["source_index"] for b in batch]).long()
    window_indices = torch.stack([b["window_index"] for b in batch]).long()

    return {
        "pose": pose_tensor,
        "gaze": gaze_tensor,
        "emotion": emotion_tensor,
        "traj": traj_tensor,
        "robot": robot_tensor,
        "modality_mask": modality_mask,
        "frame_labels": frame_labels,
        "has_pose": has_pose,
        "has_gaze": has_gaze,
        "has_emotion": has_emotion,
        "has_traj": has_traj,
        "has_robot": has_robot,
        "label": labels,
        "future_label": future_labels,
        "intent_label": intent_labels,
        "sequence_label": sequence_labels,
        "source_index": source_indices,
        "window_index": window_indices,
    }
