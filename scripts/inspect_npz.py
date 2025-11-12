"""Inspect NPZ dataset files for modality presence and shapes.

Writes a small JSON summary and prints counts. Useful to verify that the
approach-converted NPZs include pose arrays (so the loader will flag has_pose
correctly) and that future_label is present after relabeling.
"""
from pathlib import Path
import numpy as np
import json
import sys


def inspect(root: Path, max_examples=5):
    npz_files = list(root.rglob('*.npz'))
    out = {'root': str(root), 'total_files': len(npz_files), 'by_folder': {}, 'examples': []}

    for p in npz_files:
        # folder key is immediate child under root
        try:
            rel = p.relative_to(root)
            folder = rel.parts[0] if len(rel.parts) > 1 else rel.parent.name
        except Exception:
            folder = p.parent.name

        info = out['by_folder'].setdefault(folder, {'count': 0, 'has_pose': 0, 'has_traj': 0, 'has_future_label': 0})
        info['count'] += 1
        try:
            a = np.load(p)
            keys = list(a.files)
            if 'pose' in keys:
                try:
                    s = a['pose'].shape
                    if len(s) >= 2 and s[1] > 0:
                        info['has_pose'] += 1
                except Exception:
                    pass
            if 'traj' in keys:
                try:
                    s = a['traj'].shape
                    if len(s) >= 2 and s[1] > 0:
                        info['has_traj'] += 1
                except Exception:
                    pass
            if 'future_label' in keys:
                info['has_future_label'] += 1

            if len(out['examples']) < max_examples:
                ex = {'file': str(p), 'keys': keys}
                try:
                    if 'pose' in keys:
                        ex['pose_shape'] = a['pose'].shape
                    if 'traj' in keys:
                        ex['traj_shape'] = a['traj'].shape
                    if 'label' in keys:
                        ex['label_shape'] = a['label'].shape
                    if 'future_label' in keys:
                        ex['future_label_shape'] = a['future_label'].shape
                except Exception:
                    pass
                out['examples'].append(ex)

        except Exception as e:
            # note load failures
            out.setdefault('errors', []).append({'file': str(p), 'error': str(e)})

    return out


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('datasets')
    summary = inspect(root, max_examples=8)
    out_path = root / 'inspect_npz_summary.json'
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    print('Wrote summary to', out_path)
    print('Total files:', summary['total_files'])
    for k, v in summary['by_folder'].items():
        print(f"Folder: {k:30} count={v['count']:6} has_pose={v['has_pose']:6} has_traj={v['has_traj']:6} has_future_label={v['has_future_label']:6}")


if __name__ == '__main__':
    main()
