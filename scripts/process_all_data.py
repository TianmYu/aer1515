"""
Unified data processing pipeline for pedestrian intent prediction.

This script handles the complete pipeline:
1. Convert raw CSVs → NPZ format in the robot-relative frame
2. Compute future labels with specified horizon
3. Generate metadata.json with normalization stats
4. Validate all processed files

Supports two datasets:
- dataverse (Lobby_1, Lobby_2): keypoint pose + trajectory, aligned to robot base pose
- approach (Dataset-approach): motion features + trajectory, aligned to tracked robot anchor

Usage:
    # Process dataverse dataset (keypoint pose)
    python scripts/process_all_data.py --dataset dataverse --horizon 4.0 --output datasets/npz_h4s
    
    # Process approach dataset (motion features, traj-only)
    python scripts/process_all_data.py --dataset approach --horizon 4.0 --output datasets/npz_approach_h4s --mm-to-m --traj-only
    
    # Process approach dataset (motion features included)
    python scripts/process_all_data.py --dataset approach --horizon 4.0 --output datasets/npz_approach_h4s --mm-to-m
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DataProcessor:
    """Handles conversion, labeling, and validation of pedestrian datasets."""
    
    def __init__(self, dataset_type: str, output_dir: Path, horizon_sec: float = 4.0, 
                 sample_rate: float = 5.0, mm_to_m: bool = False, traj_only: bool = False):
        self.dataset_type = dataset_type
        self.output_dir = output_dir
        self.horizon_frames = int(round(horizon_sec * sample_rate))
        self.sample_rate = sample_rate
        self.mm_to_m = mm_to_m
        self.traj_only = traj_only
        
        # Statistics for metadata
        self.total_files = 0
        self.total_frames = 0
        self.traj_sum = None
        self.traj_sumsq = None
        self.pose_dims = set()
        
    def process_dataverse(self, input_dir: Path) -> int:
        """Process dataverse dataset (Lobby_1, Lobby_2 with keypoint pose data)."""
        print(f"\n=== Processing Dataverse Dataset ===")
        print(f"Input: {input_dir}")
        print(f"Output: {self.output_dir}")
        
        # Find all processed CSV files
        csv_files = list(input_dir.rglob("*.csv"))
        if not csv_files:
            raise RuntimeError(f"No CSV files found in {input_dir}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_path in csv_files:
            try:
                self._process_dataverse_file(csv_path)
            except Exception as e:
                print(f"ERROR processing {csv_path}: {e}")
        
        return self.total_files
    
    def _process_dataverse_file(self, csv_path: Path):
        """Process a single dataverse CSV file."""
        df = pd.read_csv(csv_path)

        robot_x = df.get('robot_base_x')
        robot_y = df.get('robot_base_y')
        robot_yaw = df.get('robot_base_yaw')

        if robot_x is None or robot_y is None:
            # Fallback to cart pose if robot base not available
            robot_x = df.get('cart_x')
            robot_y = df.get('cart_y')
        if robot_x is None:
            robot_x = pd.Series(0.0, index=df.index)
        if robot_y is None:
            robot_y = pd.Series(0.0, index=df.index)
        if robot_yaw is None:
            robot_yaw = df.get('cart_yaw')
        if robot_yaw is None:
            robot_yaw = pd.Series(0.0, index=df.index)

        robot_x = pd.to_numeric(robot_x, errors='coerce').ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
        robot_y = pd.to_numeric(robot_y, errors='coerce').ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
        robot_yaw = pd.to_numeric(robot_yaw, errors='coerce').ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)

        cos_yaw = np.cos(robot_yaw).astype(np.float32)
        sin_yaw = np.sin(robot_yaw).astype(np.float32)

        # Extract pose columns (*_x and *_y for keypoints)
        pose_cols = sorted([c for c in df.columns if c.endswith('_x') or c.endswith('_y')])

        ignore_prefixes = ('robot_base_', 'cart_', 'camera_')
        coord_pairs = []
        for col in pose_cols:
            if col.endswith('_x'):
                base = col[:-2]
                if any(col.startswith(prefix) for prefix in ignore_prefixes):
                    continue
                mate = f"{base}_y"
                if mate in df.columns:
                    coord_pairs.append((col, mate))

        pose_arrays: List[np.ndarray] = []
        for x_col, y_col in coord_pairs:
            x_vals = pd.to_numeric(df[x_col], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
            y_vals = pd.to_numeric(df[y_col], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
            rel_x, rel_y = self._translate_and_rotate(x_vals, y_vals, robot_x, robot_y, cos_yaw, sin_yaw)
            pose_arrays.append(rel_x)
            pose_arrays.append(rel_y)

        if pose_arrays:
            pose = np.column_stack(pose_arrays).astype(np.float32)
        else:
            pose = np.zeros((len(df), 0), dtype=np.float32)

        # Extract trajectory (pelvis_x/y or cart_x/y)
        if 'pelvis_x' in df.columns and 'pelvis_y' in df.columns:
            traj_x = pd.to_numeric(df['pelvis_x'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
            traj_y = pd.to_numeric(df['pelvis_y'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        elif 'cart_x' in df.columns and 'cart_y' in df.columns:
            traj_x = pd.to_numeric(df['cart_x'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
            traj_y = pd.to_numeric(df['cart_y'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        else:
            traj_x = np.zeros(len(df), dtype=np.float32)
            traj_y = np.zeros(len(df), dtype=np.float32)

        traj_rel_x, traj_rel_y = self._translate_and_rotate(traj_x, traj_y, robot_x, robot_y, cos_yaw, sin_yaw)
        traj = np.column_stack([traj_rel_x, traj_rel_y]).astype(np.float32)

        # Extract current labels
        if 'interacting' in df.columns:
            label = df['interacting'].fillna(False).astype(np.uint8).to_numpy()
        else:
            label = np.zeros(len(df), dtype=np.uint8)

        # Compute future labels with horizon
        future_label = self._compute_future_label(label, self.horizon_frames)

        # Save NPZ file
        out_path = self.output_dir / (csv_path.stem + '.npz')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            out_path,
            pose=pose,
            traj=traj,
            label=label,
            future_label=future_label
        )

        # Update statistics
        self._update_stats(pose, traj)
        self.total_files += 1

        if self.total_files % 100 == 0:
            print(f"  Processed {self.total_files} files...")
    
    def process_approach(self, input_dir: Path) -> int:
        """Process Dataset-approach (motion features + trajectory)."""
        print(f"\n=== Processing Approach Dataset ===")
        print(f"Input: {input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"MM to M conversion: {self.mm_to_m}")
        print(f"Trajectory-only mode: {self.traj_only}")
        
        csv_files = list(input_dir.rglob("*.csv"))
        if not csv_files:
            raise RuntimeError(f"No CSV files found in {input_dir}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_path in csv_files:
            try:
                self._process_approach_file(csv_path)
            except Exception as e:
                print(f"ERROR processing {csv_path}: {e}")
        
        return self.total_files
    
    def _process_approach_file(self, csv_path: Path):
        """Process a single approach dataset CSV file."""
        # Determine session label from folder structure
        parts_lower = [str(p).lower() for p in csv_path.parts]
        if any('intentiontointeract' in p for p in parts_lower):
            session_label = 1
        elif any('otherdistinctiveintention' in p for p in parts_lower):
            session_label = 0
        else:
            session_label = 0
        
        # Read CSV (handle commented headers)
        df = self._read_approach_csv(csv_path)
        
        if 'id' not in df.columns:
            if 'uniqueID' in df.columns:
                df = df.rename(columns={'uniqueID': 'id'})
            else:
                print(f"  Skipping {csv_path} (no person ID column)")
                return
        
        # Sort by time if available
        if 'time' in df.columns:
            df = df.sort_values('time')
        
        robot_track = df[df['type'] == 1].copy()
        if not robot_track.empty:
            robot_track = robot_track[['time', 'x', 'y', 'mTheta']]
            for col in ['time', 'x', 'y', 'mTheta']:
                robot_track[col] = pd.to_numeric(robot_track[col], errors='coerce')
            robot_track = robot_track.dropna(subset=['time']).sort_values('time')
        else:
            robot_track = pd.DataFrame(columns=['time', 'x', 'y', 'mTheta'])

        # Process each person separately
        for pid, group_df in df[df['type'] == 0].groupby('id'):
            self._process_approach_person(group_df, str(pid), session_label, csv_path, robot_track)
    
    def _read_approach_csv(self, csv_path: Path) -> pd.DataFrame:
        """Read approach CSV with commented headers."""
        default_cols = ['time', 'idx', 'id', 'type', 'x', 'y', 'z', 'vel', 
                       'mTheta', 'oTheta', 'head', 'uniqueID', 'options']
        
        # Find header line
        cols = None
        try:
            with csv_path.open('r', encoding='utf-8', errors='ignore') as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith('#') and (',' in line):
                        cols = [c.strip() for c in line.lstrip('#').strip().split(',') if c.strip()]
                        break
        except Exception:
            pass
        
        if cols is None:
            cols = default_cols
        
        # Read CSV skipping comment lines
        try:
            df = pd.read_csv(csv_path, comment='#', header=None, names=cols, low_memory=False)
        except Exception:
            df = pd.read_csv(csv_path, low_memory=False)
        
        return df
    
    def _translate_and_rotate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        origin_x: np.ndarray,
        origin_y: np.ndarray,
        cos_yaw: np.ndarray,
        sin_yaw: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Translate by robot position and rotate into robot heading frame."""
        dx = x - origin_x
        dy = y - origin_y
        rel_x = cos_yaw * dx + sin_yaw * dy
        rel_y = -sin_yaw * dx + cos_yaw * dy
        return rel_x.astype(np.float32), rel_y.astype(np.float32)

    def _align_robot_track(
        self,
        times: np.ndarray,
        robot_track: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return robot pose arrays aligned with provided timestamps."""
        if robot_track.empty:
            zeros = np.zeros_like(times, dtype=np.float32)
            return zeros, zeros, zeros

        robot_times = robot_track['time'].to_numpy(dtype=np.float64)
        robot_x = robot_track['x'].to_numpy(dtype=np.float32)
        robot_y = robot_track['y'].to_numpy(dtype=np.float32)
        robot_theta = robot_track['mTheta'].to_numpy(dtype=np.float32)

        times = np.nan_to_num(times, nan=robot_times[0] if robot_times.size else 0.0)

        idx = np.searchsorted(robot_times, times, side='left')
        idx = np.clip(idx, 0, len(robot_times) - 1)
        prev_idx = np.clip(idx - 1, 0, len(robot_times) - 1)

        prev_dist = np.abs(times - robot_times[prev_idx])
        next_dist = np.abs(times - robot_times[idx])
        choose_prev = prev_dist <= next_dist
        nearest_idx = np.where(choose_prev, prev_idx, idx)

        return robot_x[nearest_idx], robot_y[nearest_idx], robot_theta[nearest_idx]

    def _wrap_angle(self, angle: np.ndarray) -> np.ndarray:
        """Wrap angles to [-pi, pi]."""
        wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
        return wrapped.astype(np.float32)

    def _process_approach_person(self, group_df: pd.DataFrame, pid: str, 
                                 session_label: int, csv_path: Path, robot_track: pd.DataFrame):
        """Process data for a single person in approach dataset."""
        # Ensure required columns exist
        required = ['x', 'y']
        if not all(col in group_df.columns for col in required):
            return
        
        # Extract and prepare columns
        cols_to_extract = ['time', 'x', 'y', 'z', 'vel', 'mTheta', 'oTheta', 'head']
        g = group_df[[c for c in cols_to_extract if c in group_df.columns]].copy()
        
        # Convert to numeric
        for col in g.columns:
            g[col] = pd.to_numeric(g[col], errors='coerce').fillna(0.0)
        
        # Ensure all expected columns exist
        for col in ['z', 'vel', 'mTheta', 'oTheta', 'head']:
            if col not in g.columns:
                g[col] = 0.0
        
        if len(g) == 0:
            return
        
        # Build pose array (motion features) in robot-relative frame
        times = g['time'].to_numpy(dtype=np.float64)
        robot_x, robot_y, robot_theta = self._align_robot_track(times, robot_track)

        human_x = g['x'].to_numpy(dtype=np.float32)
        human_y = g['y'].to_numpy(dtype=np.float32)

        if self.mm_to_m:
            human_x = human_x / 1000.0
            human_y = human_y / 1000.0
            robot_x = robot_x / 1000.0
            robot_y = robot_y / 1000.0

        cos_theta = np.cos(robot_theta).astype(np.float32)
        sin_theta = np.sin(robot_theta).astype(np.float32)
        traj_rel_x, traj_rel_y = self._translate_and_rotate(human_x, human_y, robot_x, robot_y, cos_theta, sin_theta)
        traj = np.column_stack([traj_rel_x, traj_rel_y]).astype(np.float32)

        if self.traj_only:
            pose = np.zeros((len(g), 0), dtype=np.float32)
        else:
            vel = g['vel'].to_numpy(dtype=np.float32)
            if self.mm_to_m:
                vel = vel / 1000.0

            rel_mtheta = self._wrap_angle(g['mTheta'].to_numpy(dtype=np.float32) - robot_theta)
            rel_otheta = self._wrap_angle(g['oTheta'].to_numpy(dtype=np.float32) - robot_theta)
            rel_head = self._wrap_angle(g['head'].to_numpy(dtype=np.float32) - robot_theta)

            pose = np.column_stack([
                vel,
                rel_mtheta,
                rel_otheta,
                rel_head
            ]).astype(np.float32)
        
        # Session-level labels (all frames get same label)
        label = np.full(len(g), session_label, dtype=np.uint8)
        
        # Compute future labels
        future_label = self._compute_future_label(label, self.horizon_frames)
        
        # Save NPZ file
        rel_path = csv_path.relative_to(csv_path.parents[3])  # Preserve folder structure
        out_dir = self.output_dir / rel_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = out_dir / f"{csv_path.stem}_pid{pid}.npz"
        
        np.savez_compressed(
            out_path,
            pose=pose,
            traj=traj,
            label=label,
            future_label=future_label
        )
        
        # Update statistics
        self._update_stats(pose, traj)
        self.total_files += 1
        
        if self.total_files % 100 == 0:
            print(f"  Processed {self.total_files} files...")
    
    def _compute_future_label(self, label: np.ndarray, horizon: int) -> np.ndarray:
        """Compute future labels: will any frame in next H frames be positive?"""
        n = len(label)
        future = np.zeros(n, dtype=np.uint8)
        
        if n == 0 or horizon <= 0:
            return future
        
        # Use cumulative sum for efficient computation
        cumsum = np.concatenate([[0], np.cumsum(label.astype(np.int32))])
        
        for i in range(n):
            end_idx = min(n, i + horizon + 1)
            # Sum over frames (i+1) to (i+horizon)
            future_sum = cumsum[end_idx] - cumsum[i + 1]
            future[i] = 1 if future_sum > 0 else 0
        
        return future
    
    def _update_stats(self, pose: np.ndarray, traj: np.ndarray):
        """Update running statistics for metadata."""
        n_frames = len(traj)
        self.total_frames += n_frames
        
        if pose.shape[1] > 0:
            self.pose_dims.add(pose.shape[1])
        
        # Update trajectory stats
        if self.traj_sum is None:
            self.traj_sum = np.zeros(traj.shape[1], dtype=np.float64)
            self.traj_sumsq = np.zeros(traj.shape[1], dtype=np.float64)
        
        self.traj_sum += traj.sum(axis=0)
        self.traj_sumsq += (traj ** 2).sum(axis=0)
    
    def write_metadata(self):
        """Write metadata.json with normalization statistics."""
        if self.total_frames == 0:
            print("WARNING: No frames processed, skipping metadata")
            return
        
        if self.traj_sum is None or self.traj_sumsq is None:
            print("WARNING: Missing trajectory statistics, metadata incomplete")
            return

        # Compute trajectory mean and std
        traj_mean = (self.traj_sum / self.total_frames).tolist()
        traj_var = (self.traj_sumsq / self.total_frames - np.array(traj_mean) ** 2)
        traj_std = np.sqrt(np.maximum(traj_var, 0.0)).tolist()
        
        metadata = {
            'dataset_type': self.dataset_type,
            'total_files': self.total_files,
            'total_frames': self.total_frames,
            'horizon_frames': self.horizon_frames,
            'horizon_seconds': self.horizon_frames / self.sample_rate,
            'sample_rate_hz': self.sample_rate,
            'traj_mean': traj_mean,
            'traj_std': traj_std,
            'pose_dimensions': sorted(list(self.pose_dims)),
            'traj_only': self.traj_only,
            'mm_to_m_conversion': self.mm_to_m,
            'coordinate_frame': 'robot_relative'
        }
        
        meta_path = self.output_dir / 'metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n=== Metadata Written ===")
        print(f"Location: {meta_path}")
        print(f"Total files: {self.total_files}")
        print(f"Total frames: {self.total_frames}")
        print(f"Trajectory mean: {traj_mean}")
        print(f"Trajectory std: {traj_std}")
        print(f"Pose dimensions: {sorted(list(self.pose_dims))}")
    
    def validate(self) -> bool:
        """Validate all processed NPZ files."""
        print(f"\n=== Validating Processed Data ===")
        
        npz_files = list(self.output_dir.rglob("*.npz"))
        if not npz_files:
            print("ERROR: No NPZ files found!")
            return False
        
        print(f"Found {len(npz_files)} NPZ files")
        
        errors = []
        label_pos_count = 0
        label_neg_count = 0
        future_pos_count = 0
        future_neg_count = 0
        
        for npz_path in npz_files:
            try:
                data = np.load(npz_path)
                
                # Check required keys
                required_keys = ['pose', 'traj', 'label', 'future_label']
                for key in required_keys:
                    if key not in data:
                        errors.append(f"{npz_path.name}: missing key '{key}'")
                        continue
                
                pose = data['pose']
                traj = data['traj']
                label = data['label']
                future_label = data['future_label']
                
                # Check shapes
                n = len(traj)
                if len(pose) != n:
                    errors.append(f"{npz_path.name}: pose length {len(pose)} != traj length {n}")
                if len(label) != n:
                    errors.append(f"{npz_path.name}: label length {len(label)} != traj length {n}")
                if len(future_label) != n:
                    errors.append(f"{npz_path.name}: future_label length {len(future_label)} != traj length {n}")
                
                # Check trajectory shape
                if traj.shape[1] < 2:
                    errors.append(f"{npz_path.name}: trajectory has only {traj.shape[1]} columns (need at least 2)")
                
                # Count labels
                label_pos_count += int(label.sum())
                label_neg_count += int((label == 0).sum())
                future_pos_count += int(future_label.sum())
                future_neg_count += int((future_label == 0).sum())
                
            except Exception as e:
                errors.append(f"{npz_path.name}: {str(e)}")
        
        # Print results
        if errors:
            print(f"\nFound {len(errors)} errors:")
            for err in errors[:20]:  # Print first 20 errors
                print(f"  - {err}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more")
            return False
        
        print("\n✓ All files validated successfully!")
        print(f"\nLabel distribution:")
        print(f"  Current labels:  {label_pos_count:6d} positive, {label_neg_count:6d} negative "
              f"({100*label_pos_count/(label_pos_count+label_neg_count):.1f}% positive)")
        print(f"  Future labels:   {future_pos_count:6d} positive, {future_neg_count:6d} negative "
              f"({100*future_pos_count/(future_pos_count+future_neg_count):.1f}% positive)")
        
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, 
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--dataset', choices=['dataverse', 'approach'], required=True,
                       help='Dataset to process: dataverse (keypoint) or approach (motion)')
    parser.add_argument('--output', required=True,
                       help='Output directory for processed NPZ files')
    parser.add_argument('--horizon', type=float, default=4.0,
                       help='Future label horizon in seconds (default: 4.0)')
    parser.add_argument('--sample-rate', type=float, default=5.0,
                       help='Dataset sample rate in Hz (default: 5.0)')
    parser.add_argument('--mm-to-m', action='store_true',
                       help='Convert millimeters to meters (for approach dataset)')
    parser.add_argument('--traj-only', action='store_true',
                       help='Save trajectory only, no pose features (sets has_pose=False)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip validation step')
    
    args = parser.parse_args()
    
    # Set up paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create processor
    processor = DataProcessor(
        dataset_type=args.dataset,
        output_dir=output_dir,
        horizon_sec=args.horizon,
        sample_rate=args.sample_rate,
        mm_to_m=args.mm_to_m,
        traj_only=args.traj_only
    )
    
    # Process dataset
    if args.dataset == 'dataverse':
        input_dir = Path('datasets/dataverse_files')
        if not input_dir.exists():
            raise RuntimeError(f"Input directory not found: {input_dir}")
        processor.process_dataverse(input_dir)
    
    elif args.dataset == 'approach':
        input_dir = Path('datasets/Dataset-approach/Dataset')
        if not input_dir.exists():
            raise RuntimeError(f"Input directory not found: {input_dir}")
        processor.process_approach(input_dir)
    
    # Write metadata
    processor.write_metadata()
    
    # Validate
    if not args.skip_validation:
        valid = processor.validate()
        if not valid:
            print("\n❌ Validation failed! Please check errors above.")
            return 1
    
    print(f"\n✅ Processing complete! Output: {output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())
