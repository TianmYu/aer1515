"""Preprocess Shutter/Dataverse CSVs into per-person, pre-interaction tracks.

This script does the following for each CSV under an input directory:
- Reads the CSV into a DataFrame
- Groups rows by `seq_id` (per-person track)
- For each seq_id: finds the first frame where the person is interacting (by checking
  a set of interaction-related columns). It drops that frame and any rows after it
  (so we only keep timesteps before interaction begins).
- Normalizes/cleans boolean columns such as `interacting`, `currently_interacting`,
  `has_interacted`, `will_interact`, `future_interaction` into proper booleans.
- Exports one CSV per (original file, seq_id) into the output directory. The
  output filename is: <origname>__seq-<seq_id>.csv

This makes it easier for training code to treat each person independently and
ensures windows won't contain frames after interaction started (which are not
useful for predicting intent beforehand).

Usage (from project root):
  python scripts/preprocess_tracks.py --input datasets/dataverse_files --output datasets/processed --max-files 20

"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


def parse_bool_series(s):
    """Robustly parse a series of booleans which may be bools, strings, or numeric.
    Returns a Series of dtype pd.BooleanDtype() (nullable boolean).
    """
    if s.dtype == bool:
        return s.astype('boolean')
    # Try common string values
    def _parse(x):
        if pd.isna(x):
            return pd.NA
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if isinstance(x, (int, np.integer)):
            return bool(x)
        tx = str(x).strip().lower()
        if tx in ('true', 't', '1', 'yes', 'y'):
            return True
        if tx in ('false', 'f', '0', 'no', 'n'):
            return False
        return pd.NA
    return s.map(_parse).astype('boolean')


def first_interaction_index(df, cols):
    """Return the integer index (position) of the first row where any of the
    columns in `cols` is True. If none are True, return None."""
    for c in cols:
        if c not in df.columns:
            continue
        series = parse_bool_series(df[c])
        mask = series.fillna(False).to_numpy()
        where = np.nonzero(mask)[0]
        if where.size > 0:
            return int(where[0])
    return None


def process_file(path: Path, out_dir: Path, trim_post_interaction: bool = True, mark_seq_interacted: bool = False, interaction_cols=('interacting','currently_interacting','has_interacted','will_interact','future_interaction')):
    df = pd.read_csv(path)

    if 'seq_id' not in df.columns:
        # If there's no seq_id, treat whole file as one sequence
        df['_seq_id_fallback'] = 0
        seq_col = '_seq_id_fallback'
    else:
        seq_col = 'seq_id'

    saved = 0
    for seq_id, g in df.groupby(seq_col):
        g = g.reset_index(drop=True)
        # Clean boolean columns
        for c in interaction_cols:
            if c in g.columns:
                g[c] = parse_bool_series(g[c])

        # find first interaction across any of the interaction columns (for bookkeeping)
        idx = first_interaction_index(g, interaction_cols)
        if trim_post_interaction:
            # keep only rows strictly BEFORE interaction start (original behavior)
            if idx is not None:
                g_pre = g.iloc[:idx].copy()
            else:
                g_pre = g.copy()
        else:
            # keep full sequence
            g_pre = g.copy()

        if len(g_pre) == 0:
            # nothing to save for this seq
            continue

        # Ensure a canonical future_interaction boolean column exists.
        # Two modes supported:
        # - mark_seq_interacted: if True, then if this seq contains any interaction
        #   set future_interaction=True for all rows in the (kept) sequence.
        # - otherwise, prefer existing columns 'future_interaction' or 'will_interact',
        #   or compute a per-row future flag (whether any later row has interacting==True).
        if mark_seq_interacted:
            seq_has_interaction = idx is not None
            g_pre['future_interaction'] = pd.Series([seq_has_interaction] * len(g_pre), dtype='boolean')
        else:
            if 'future_interaction' in g_pre.columns:
                g_pre['future_interaction'] = parse_bool_series(g_pre['future_interaction'])
            elif 'will_interact' in g_pre.columns:
                g_pre['future_interaction'] = parse_bool_series(g_pre['will_interact'])
            else:
                # compute per-row: whether any later row in the original full sequence has interacting==True
                # build a combined interacting mask from the original sequence g (not g_pre)
                combined = np.zeros(len(g), dtype=bool)
                for c in interaction_cols:
                    if c in g.columns:
                        combined = combined | g[c].fillna(False).astype(bool).to_numpy()
                # for each index in g_pre, check if any later index in g has interaction True
                future_mask = []
                for i in range(len(g_pre)):
                    if i + 1 < len(combined):
                        future_mask.append(bool(combined[i+1:].any()))
                    else:
                        future_mask.append(False)
                g_pre['future_interaction'] = pd.Series(future_mask, dtype='boolean')

        # Output path
        stem = path.stem
        out_name = f"{stem}__seq-{seq_id}.csv"
        out_path = out_dir / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        g_pre.to_csv(out_path, index=False)
        saved += 1

    return saved


def main():
    p = argparse.ArgumentParser(description="Preprocess shutter/dataverse CSVs into per-person pre-interaction tracks")
    p.add_argument('--input', '-i', type=str, default='datasets/dataverse_files', help='Input directory to recursively search for CSVs')
    p.add_argument('--output', '-o', type=str, default='datasets/processed', help='Output directory to write per-person CSVs')
    p.add_argument('--max-files', type=int, default=None, help='Process at most N input CSV files (useful for quick checks)')
    p.add_argument('--ext', type=str, default='csv', help='CSV file extension to match')
    p.add_argument('--no-trim', dest='no_trim', action='store_true', help='Do NOT trim frames after interaction start (keep full sequences)')
    p.add_argument('--mark-seq-interacted', dest='mark_seq_interacted', action='store_true', help='If set, mark entire sequence as future_interaction=True when any interaction occurs')
    p.set_defaults(no_trim=False, mark_seq_interacted=False)
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    csvs = list(inp.rglob(f'*.{args.ext}'))
    if args.max_files:
        csvs = csvs[:args.max_files]

    total_saved = 0
    for pth in tqdm(csvs, desc='files'):
        try:
            saved = process_file(pth, out, trim_post_interaction=not args.no_trim, mark_seq_interacted=args.mark_seq_interacted)
            total_saved += saved
        except Exception as e:
            print(f"Failed to process {pth}: {e}")

    print(f"Done. Processed {len(csvs)} files and saved {total_saved} per-person CSVs under {out}")


if __name__ == '__main__':
    main()
