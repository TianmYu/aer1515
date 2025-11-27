"""Data processing utilities consolidated for the gaze project.

Contains functions for:
- parsing Columbia filenames -> angles
- converting angles -> unit 3D gaze vectors
- writing p10-style annotation files from Columbia images
- fixing annotation path prefixes
- splitting annotations by subject (per-subject holdout)
- collecting annotation lines from p10-style files or directories

This central file reduces duplication and provides toggles for behavior.
"""
import os
import re
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


def is_mpii_gaze_image(path_or_str) -> bool:
    """Check if an image path belongs to MPIIGaze dataset.
    
    MPIIGaze uses opposite coordinate convention from Columbia dataset,
    requiring axis flips for gaze vectors.
    """
    path_str = str(path_or_str)
    # Exclude Columbia dataset even though it's under MPIIGazeSet directory
    if 'columbia' in path_str.lower():
        return False
    return 'MPIIGaze' in path_str or 'mpiigaze' in path_str.lower()


def apply_dataset_coordinate_transform(gaze_vec: np.ndarray, path_or_str) -> np.ndarray:
    """Apply coordinate system transform to convert annotations to display space.
    
    Display convention: +X=right, +Y=up, +Z=forward
    
    - MPIIGaze annotations: use -X for right, +Y for up -> flip X only
    - Columbia annotations: use +X for right, +Y for up -> NO transform needed
    
    Args:
        gaze_vec: 3D gaze vector (x, y, z)
        path_or_str: image path to determine dataset
        
    Returns:
        Transformed gaze vector
    """
    vec = gaze_vec.copy()
    if is_mpii_gaze_image(path_or_str):
        vec[0] = -vec[0]  # flip X: MPIIGaze uses -X for right
    # Columbia needs no transform - already in correct convention
    return vec


def parse_angles_from_name(name: str) -> Optional[Tuple[float, float]]:
    m_h = re.search(r'([+-]?\d+)H', name, flags=re.IGNORECASE)
    m_v = re.search(r'([+-]?\d+)V', name, flags=re.IGNORECASE)
    if not m_h or not m_v:
        return None
    try:
        h = float(m_h.group(1))
        v = float(m_v.group(1))
        return h, v
    except ValueError:
        return None


def angles_to_unit(h_deg: float, v_deg: float) -> Optional[Tuple[float, float, float]]:
    yaw = math.radians(h_deg)
    pitch = math.radians(v_deg)
    x = math.sin(yaw) * math.cos(pitch)
    y = math.sin(pitch)
    z = math.cos(yaw) * math.cos(pitch)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0:
        return None
    return x / norm, y / norm, z / norm


def convert_columbia_to_p10(
    base_mpiigaze_dir: str,
    out_path: str,
    depth: float = 1.0,
    label: str = 'columbia',
    write_fc_zero: bool = True,
):
    base_mpiigaze_dir = str(base_mpiigaze_dir)
    col_dir = os.path.join(base_mpiigaze_dir, 'Columbia Gaze Data Set')
    if not os.path.isdir(col_dir):
        # try alternate name without spaces
        col_dir = os.path.join(base_mpiigaze_dir, 'Columbia_Gaze_Data_Set')
        if not os.path.isdir(col_dir):
            raise FileNotFoundError(f'Columbia directory not found under {base_mpiigaze_dir}')

    entries = []
    skipped = 0
    for root, dirs, files in os.walk(col_dir):
        files = [f for f in files if not f.startswith('.')]
        for fname in sorted(files):
            if not fname.lower().endswith(('.jpg', '.png')):
                continue
            angles = parse_angles_from_name(fname)
            rel_dir = os.path.relpath(root, base_mpiigaze_dir)
            rel_path = os.path.join(rel_dir, fname).replace('\\', '/')
            if angles is None:
                skipped += 1
                continue
            h_deg, v_deg = angles
            unit = angles_to_unit(h_deg, v_deg)
            if unit is None:
                skipped += 1
                continue
            nums = [0.0] * 26
            if not write_fc_zero:
                # placeholder: consumers can override fc tokens later
                nums[20] = 0.0
                nums[21] = 0.0
                nums[22] = 0.0
            else:
                nums[20] = 0.0
                nums[21] = 0.0
                nums[22] = 0.0
            # gaze target scaled by depth
            nums[23] = unit[0] * depth
            nums[24] = unit[1] * depth
            nums[25] = unit[2] * depth
            entries.append((rel_path, nums))

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, 'w') as f:
        for rel_path, nums in entries:
            num_strs = ['{:.6f}'.format(x) for x in nums]
            line = ' '.join([rel_path] + num_strs + [label])
            f.write(line + '\n')

    return len(entries), skipped


def fix_paths_in_file(ann_path: str, prefix: str = 'MPIIGazeSet/') -> Tuple[int, int]:
    p = Path(ann_path)
    if not p.exists():
        raise FileNotFoundError(str(ann_path))
    lines = p.read_text().splitlines()
    out = []
    changed = 0
    kept = 0
    for L in lines:
        if not L.strip():
            continue
        toks = L.split()
        path = toks[0]
        if path.startswith(prefix) or os.path.isabs(path):
            out.append(L)
            kept += 1
            continue
        new_path = prefix + path
        out.append(' '.join([new_path] + toks[1:]))
        changed += 1
    p.write_text('\n'.join(out) + '\n')
    return changed, kept


def split_annotations_by_subject(
    ann_path: str,
    out_train: str,
    out_val: str,
    val_fraction: float = 0.1,
    seed: int = 42,
    holdout_count: Optional[int] = None,
):
    p = Path(ann_path)
    if not p.exists():
        raise FileNotFoundError(str(ann_path))
    groups = {}
    for l in p.read_text().splitlines():
        if not l.strip():
            continue
        toks = l.split()
        path = toks[0]
        # Expect patterns like 'Columbia_Gaze_Data_Set/0051/...'
        parts = path.split('/')
        subj = parts[1] if len(parts) > 1 else 'unknown'
        groups.setdefault(subj, []).append(l)

    subjs = list(groups.keys())
    random.seed(seed)
    random.shuffle(subjs)
    if holdout_count is None:
        n_val = max(1, int(len(subjs) * val_fraction))
    else:
        n_val = min(len(subjs), max(0, int(holdout_count)))
    val_subjs = set(subjs[:n_val])

    with open(out_train, 'w') as ft, open(out_val, 'w') as fv:
        for s, lines in groups.items():
            if s in val_subjs:
                fv.writelines([l + '\n' for l in lines])
            else:
                ft.writelines([l + '\n' for l in lines])

    return len(subjs) - n_val, n_val


def collect_annotation_lines(path: Path) -> List[str]:
    path = Path(path)
    ann_lines: List[str] = []
    # If the path is a single annotation file, read and attempt to resolve
    # image paths intelligently. Many MPIIGaze annotation files refer to
    # relative paths like `day01/0001.jpg`; those images commonly live under
    # subfolders of a top-level `MPIIGazeSet/` directory (e.g. `p10/day01/...`).
    # Try a few heuristics to resolve such references so callers don't need to
    # pre-fix annotation files on disk.
    if path.is_file():
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        module_root = Path(__file__).resolve().parent
        mpiigaze_root = module_root / 'MPIIGazeSet'
        mpiigaze_children = []
        if mpiigaze_root.exists() and mpiigaze_root.is_dir():
            try:
                mpiigaze_children = [d for d in mpiigaze_root.iterdir() if d.is_dir()]
            except Exception:
                mpiigaze_children = []
        for L in lines:
            toks = L.split()
            if len(toks) == 0:
                continue
            orig = toks[0]
            img = Path(orig)
            resolved = None
            # 1) absolute path exists
            if img.is_absolute() and img.exists():
                resolved = str(img.resolve())
            else:
                # 2) if ann file sits next to images, resolve relative to ann file
                candidate = path.parent / img
                if candidate.exists():
                    resolved = str(candidate.resolve())
                else:
                    # 3) try direct MPIIGazeSet/<orig> from module root
                    candidate = module_root / 'MPIIGazeSet' / orig
                    if candidate.exists():
                        resolved = str(candidate.resolve())
                    else:
                        # 3b) if orig already starts with MPIIGazeSet/, try from module root
                        if orig.startswith('MPIIGazeSet/'):
                            candidate = module_root / orig
                            if candidate.exists():
                                resolved = str(candidate.resolve())
                        # 4) iterate children of MPIIGazeSet (p00..p14) and try each
                        if resolved is None:
                            for d in mpiigaze_children:
                                candidate = d / orig
                                if candidate.exists():
                                    resolved = str(candidate.resolve())
                                    break
            if resolved is None:
                # fallback: leave original token unchanged
                toks[0] = orig
            else:
                toks[0] = resolved
            ann_lines.append(' '.join(toks))
        return ann_lines

    # If the provided path is a directory, search for common MPIIGaze p_*.txt
    patterns = ['p_*.txt', 'p*.txt', 'p_.txt', 'p.txt']
    for pat in patterns:
        for f in path.rglob(pat):
            base = f.parent
            for l in [ll for ll in f.read_text().splitlines() if ll.strip()]:
                toks = l.split()
                if len(toks) == 0:
                    continue
                img = Path(toks[0])
                if not img.is_absolute():
                    toks[0] = str((base / img).resolve())
                else:
                    toks[0] = str(img.resolve())
                ann_lines.append(' '.join(toks))
    return ann_lines


def _cli_convert_columbia():
    import argparse

    p = argparse.ArgumentParser(description='Convert Columbia Gaze Data Set filenames into p10-style annotations using data_pipeline utilities')
    p.add_argument('base_dir', help='base directory that contains the Columbia dataset (parent of "Columbia Gaze Data Set" or "Columbia_Gaze_Data_Set")')
    p.add_argument('--out', '-o', dest='out_path', default='face_and_pose/MPIIGazeSet/columbia_p10.txt', help='output p10 file path')
    p.add_argument('--depth', type=float, default=1.0, help='depth (meters) to scale unit gaze vectors')
    p.add_argument('--label', default='columbia', help='label token to append to each line')
    p.add_argument('--write-fc-zero', action='store_true', dest='write_fc_zero', help='write face-center tokens as zeros (default)')
    args = p.parse_args()

    n_written, n_skipped = convert_columbia_to_p10(args.base_dir, args.out_path, depth=args.depth, label=args.label, write_fc_zero=args.write_fc_zero)
    print(f'Wrote {n_written} lines to {args.out_path} (skipped {n_skipped} files)')


if __name__ == '__main__':
    _cli_convert_columbia()
