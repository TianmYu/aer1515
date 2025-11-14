#!/usr/bin/env python3
"""Quick pose sequence visualizer for NPZ shards.

Example usage:
  python scripts/visualize_sequence.py --npz tmp_out/processed_per_person/dataverse_npz/example.npz --save demo.mp4
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text

KEYPOINT_ORDER: tuple[str, ...] = (
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
)

KP_INDEX = {name: idx for idx, name in enumerate(KEYPOINT_ORDER)}
SKELETON_EDGES: tuple[tuple[str, str], ...] = (
    ("left_ankle", "left_knee"),
    ("left_knee", "left_hip"),
    ("left_hip", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_hip", "left_shoulder"),
    ("right_hip", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
)


def _resolve_edge(idx_a: str, idx_b: str) -> tuple[int, int]:
    return KP_INDEX[idx_a], KP_INDEX[idx_b]


EDGE_INDEX = tuple(_resolve_edge(a, b) for a, b in SKELETON_EDGES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize pose sequences stored in NPZ format")
    parser.add_argument("--npz", required=True, help="Path to the sequence NPZ file")
    parser.add_argument("--out", help="Optional output video/gif path (mp4/gif)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames (default: all)")
    parser.add_argument("--fps", type=int, default=15, help="Playback frames per second")
    parser.add_argument("--title", default=None, help="Custom plot title")
    parser.add_argument("--robot-frame", action="store_true",
                        help="Reproject skeleton into the robot's frame (robot at origin, forward=+Y)")
    parser.add_argument("--show-traj", action="store_true",
                        help="Overlay pelvis trajectory (requires --robot-frame)")
    return parser.parse_args()


def load_pose_sequence(path: Path) -> tuple[
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    with np.load(path, allow_pickle=True) as data:
        pose = np.asarray(data["pose"], dtype=np.float32)
        frame_labels = np.asarray(data.get("frame_labels")) if "frame_labels" in data else None
        intent = np.asarray(data.get("intent_label")) if "intent_label" in data else None
        robot = np.asarray(data.get("robot"), dtype=np.float32) if "robot" in data else None
        traj = np.asarray(data.get("traj"), dtype=np.float32) if "traj" in data else None
    return pose, frame_labels, intent, robot, traj


def reshape_pose(pose: np.ndarray) -> np.ndarray:
    if pose.ndim != 2 or pose.shape[1] != len(KEYPOINT_ORDER) * 3:
        raise ValueError(f"Unexpected pose shape {pose.shape}; expected (T, {len(KEYPOINT_ORDER)*3})")
    coords = pose.reshape(pose.shape[0], len(KEYPOINT_ORDER), 3)
    return coords


def rotation_matrix(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def transform_to_robot_frame(
    coords: np.ndarray,
    robot_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if robot_vec.ndim != 2 or robot_vec.shape[1] < 5:
        raise ValueError("Robot modality must have shape (T, >=5) for robot-frame projection")
    transformed = coords.copy()
    root_positions = np.zeros((coords.shape[0], 2), dtype=np.float32)
    for idx in range(coords.shape[0]):
        base_dx, base_dy = float(robot_vec[idx, 0]), float(robot_vec[idx, 1])
        sin_yaw, cos_yaw = float(robot_vec[idx, 3]), float(robot_vec[idx, 4])
        # Rotate so robot forward faces +Y in the plot.
        yaw = math.atan2(sin_yaw, cos_yaw)
        rot = rotation_matrix(math.pi / 2.0 - yaw)
        offset = np.array([-base_dx, -base_dy], dtype=np.float32)
        pts = coords[idx, :, :2] + offset
        pts_rot = pts @ rot.T
        transformed[idx, :, :2] = pts_rot
        root_positions[idx] = offset @ rot.T
    transformed[:, :, 2] = coords[:, :, 2]
    return transformed, root_positions


def create_lines(ax: Axes) -> list[Line2D]:
    lines: list[Line2D] = []
    for _ in EDGE_INDEX:
        line, = ax.plot([], [], color="tab:blue", lw=2)
        lines.append(line)
    return lines


def update_frame(
    frame_idx: int,
    coords: np.ndarray,
    lines: Sequence[Line2D],
    scatter: PathCollection,
    title_artist: Text,
    frame_labels: Optional[np.ndarray],
    intent: Optional[np.ndarray],
    robot_vec: Optional[np.ndarray],
    traj_line: Optional[Line2D],
    root_positions: Optional[np.ndarray],
) -> None:
    pts = coords[frame_idx, :, :2]
    conf = coords[frame_idx, :, 2]
    scatter.set_offsets(pts)
    scatter.set_array(conf)

    for line, (src, dst) in zip(lines, EDGE_INDEX):
        line.set_data([pts[src, 0], pts[dst, 0]], [pts[src, 1], pts[dst, 1]])

    label = None
    if frame_labels is not None:
        label = int(frame_labels[frame_idx])
    intent_text = None
    if intent is not None:
        intent_text = int(np.asarray(intent).reshape(-1)[0])

    title_fragments: list[str] = [f"frame {frame_idx}"]
    if label is not None:
        title_fragments.append(f"frame_label={label}")
    if intent_text is not None:
        title_fragments.append(f"intent={intent_text}")
    if robot_vec is not None and robot_vec.shape[1] > 2:
        title_fragments.append(f"robot_dist={robot_vec[frame_idx, 2]:.2f}")
    if traj_line is not None and root_positions is not None:
        traj_line.set_data(root_positions[:frame_idx + 1, 0], root_positions[:frame_idx + 1, 1])
    title_artist.set_text(" | ".join(title_fragments))


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    pose, frame_labels, intent, robot_vec, traj = load_pose_sequence(npz_path)
    coords = reshape_pose(pose)
    total_frames = coords.shape[0]
    if args.max_frames is not None:
        total_frames = min(total_frames, args.max_frames)
        coords = coords[:total_frames]
        if frame_labels is not None:
            frame_labels = frame_labels[:total_frames]
        if robot_vec is not None:
            robot_vec = robot_vec[:total_frames]
        if traj is not None:
            traj = traj[:total_frames]

    root_positions: Optional[np.ndarray] = None
    if args.robot_frame:
        if robot_vec is None:
            raise ValueError("Robot data not present in NPZ; cannot enable --robot-frame")
        coords, root_positions = transform_to_robot_frame(coords, robot_vec)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    if not args.robot_frame:
        ax.invert_yaxis()
    pts = coords[0, :, :2]
    scatter = ax.scatter(pts[:, 0], pts[:, 1], c=coords[0, :, 2], cmap='viridis', vmin=0.0, vmax=1.0)
    lines = create_lines(ax)
    title_artist = ax.set_title(args.title or npz_path.name)
    if args.robot_frame:
        ax.set_xlabel('lateral (robot frame)')
        ax.set_ylabel('forward (robot frame)')
    else:
        ax.set_xlabel('x (root-normalised)')
        ax.set_ylabel('y (root-normalised)')

    heading_line: Optional[Line2D] = None
    traj_line: Optional[Line2D] = None
    robot_marker = None
    if args.robot_frame:
        robot_marker = ax.scatter([0.0], [0.0], marker='x', s=90, c='red', label='robot')
        heading_line, = ax.plot([0.0, 0.0], [0.0, 1.0], color='red', lw=1.5, label='forward')
        if args.show_traj and root_positions is not None:
            traj_line, = ax.plot([], [], color='black', lw=1.0, ls='--', alpha=0.5, label='pelvis path')
        if robot_marker or heading_line or traj_line:
            ax.legend(loc='upper right')

    # Auto-scale axes for current view.
    span = np.nanmax(np.abs(coords[:, :, :2]))
    if np.isfinite(span):
        margin = 0.5 if args.robot_frame else 0.2
        ax.set_xlim(-span - margin, span + margin)
        ax.set_ylim(-span - margin, span + margin)

    def _step(idx: int) -> Iterable:
        update_frame(
            idx,
            coords,
            lines,
            scatter,
            title_artist,
            frame_labels,
            intent,
            robot_vec,
            traj_line,
            root_positions,
        )
        extras = [scatter, title_artist]
        if traj_line is not None:
            extras.append(traj_line)
        if heading_line is not None:
            extras.append(heading_line)
        if robot_marker is not None:
            extras.append(robot_marker)
        return itertools.chain(lines, extras)

    anim = animation.FuncAnimation(fig, _step, frames=range(total_frames), interval=1000.0 / max(args.fps, 1), blit=False)

    if args.out:
        out_path = Path(args.out)
        writer: animation.AbstractMovieWriter
        if out_path.suffix.lower() == '.gif':
            writer = animation.PillowWriter(fps=args.fps)
        else:
            writer = animation.FFMpegWriter(fps=args.fps)
        print(f'Saving animation to {out_path}')
        anim.save(str(out_path), writer=writer)
    else:
        plt.show()


if __name__ == "__main__":
    main()
