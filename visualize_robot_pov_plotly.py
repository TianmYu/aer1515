#!/usr/bin/env python3
"""
visualize_robot_pov_plotly.py

Interactive Plotly visualization for robot-perspective CSV records of people.

Generates:
 - interactive HTML trajectory (robot-centric) with time color and hover info
 - interactive time-series HTML (distance, speed, will_interact)
 - interactive distance vs speed scatter HTML
 - optional lightweight animation HTML (person marker +/- keypoints per frame)

Usage:
  python visualize_robot_pov_plotly.py /path/to/csv.csv --outdir viz_plotly_out
  python visualize_robot_pov_plotly.py csv.csv --outdir viz_out --person-anchor pelvis --robot-anchor robot_base --no-animate

Dependencies:
  pip install pandas numpy plotly

Notes:
 - Script auto-expands bracketed '[x y z]' columns into col_x/col_y/col_z when present.
 - It auto-detects person anchor (pelvis/head/neck/spine_navel) and robot anchor (robot_base/robot_head).
 - If auto-detect fails pass --person-anchor and --robot-anchor (prefixes without _x/_y).
 - Outputs are saved as HTML files in --outdir.
"""

import argparse
import ast
import os
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- helpers ----------
def parse_bracket_array(s: str):
    if pd.isna(s):
        return None
    s = str(s).strip()
    if s == "":
        return None
    try:
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            parts = [p for p in inner.replace(",", " ").split() if p != ""]
            return np.array([float(p) for p in parts], dtype=float)
    except Exception:
        pass
    try:
        val = ast.literal_eval(s)
        return np.array(val, dtype=float)
    except Exception:
        return None


def load_csv(path: str, verbose: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if verbose:
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    # expand bracket fields into X/Y/Z
    for col in list(df.columns):
        sample = df[col].astype(str).head(200).str.strip()
        if sample.str.match(r"^\[.*\]$").any():
            parsed = df[col].apply(parse_bracket_array)
            lens = parsed.dropna().map(len) if parsed.dropna().size > 0 else pd.Series(dtype=int)
            max_len = int(lens.max()) if not lens.empty else 0
            for i, axis in enumerate(["x", "y", "z"][:max_len]):
                new_col = f"{col}_{axis}"
                df[new_col] = parsed.apply(lambda arr: float(arr[i]) if isinstance(arr, (list, np.ndarray)) and len(arr) > i else np.nan)
            print(f"Expanded '{col}' -> {[f'{col}_{a}' for a in ['x','y','z'][:max_len]]}")
    # try cast obvious numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass
    return df


def choose_columns(df: pd.DataFrame, person_anchor: Optional[str] = None, robot_anchor: Optional[str] = None) -> Dict[str, str]:
    cols_lower = {c.lower(): c for c in df.columns}
    mapping = {}
    # time
    for t in ["resampled_timestamp", "timestamp", "skel_timestamp", "tf_timestamp"]:
        if t in cols_lower:
            mapping["time"] = cols_lower[t]
            break
    # person anchor
    if person_anchor:
        px = f"{person_anchor}_x"
        py = f"{person_anchor}_y"
        if px in cols_lower and py in cols_lower:
            mapping["person_x"] = cols_lower[px]
            mapping["person_y"] = cols_lower[py]
    else:
        for p in ["pelvis_x", "head_x", "spine_navel_x", "spine_chest_x", "neck_x", "nose_x"]:
            if p in cols_lower:
                mapping["person_x"] = cols_lower[p]
                mapping["person_y"] = cols_lower[p.replace("_x", "_y")]
                break
    # robot anchor
    if robot_anchor:
        rx = f"{robot_anchor}_x"
        ry = f"{robot_anchor}_y"
        if rx in cols_lower and ry in cols_lower:
            mapping["robot_x"] = cols_lower[rx]
            mapping["robot_y"] = cols_lower[ry]
    else:
        for r in ["robot_base_x", "robot_head_x", "robot_base_x"]:
            if r in cols_lower:
                mapping["robot_x"] = cols_lower[r]
                mapping["robot_y"] = cols_lower[r.replace("_x", "_y")]
                break
    return mapping


def ensure_time(df: pd.DataFrame, time_col: Optional[str]):
    if time_col and time_col in df.columns:
        t = pd.to_numeric(df[time_col], errors="coerce")
        if (t > 1e9).any():
            return (t - t.iloc[0]).astype(float)
        else:
            return (t - t.iloc[0]).astype(float)
    else:
        return pd.Series(np.arange(len(df)), index=df.index, name="time")


def to_robot_frame(df: pd.DataFrame, person_x: str, person_y: str, robot_x: str, robot_y: str) -> Tuple[np.ndarray, np.ndarray]:
    px = pd.to_numeric(df[person_x], errors="coerce").to_numpy(dtype=float)
    py = pd.to_numeric(df[person_y], errors="coerce").to_numpy(dtype=float)
    rx = pd.to_numeric(df[robot_x], errors="coerce").to_numpy(dtype=float)
    ry = pd.to_numeric(df[robot_y], errors="coerce").to_numpy(dtype=float)
    rel_x = px - rx
    rel_y = py - ry
    return rel_x, rel_y


def compute_speed(df: pd.DataFrame, person_x: str, person_y: str, time: pd.Series):
    if "x_vel" in df.columns and "y_vel" in df.columns:
        vx = pd.to_numeric(df["x_vel"], errors="coerce").to_numpy()
        vy = pd.to_numeric(df["y_vel"], errors="coerce").to_numpy()
        return np.sqrt(np.nan_to_num(vx) ** 2 + np.nan_to_num(vy) ** 2)
    # try person_velocity
    if "person_velocity" in df.columns:
        return pd.to_numeric(df["person_velocity"], errors="coerce").to_numpy()
    # fallback: finite-difference on anchor
    px = pd.to_numeric(df[person_x], errors="coerce").to_numpy()
    py = pd.to_numeric(df[person_y], errors="coerce").to_numpy()
    t = time.to_numpy()
    # prevent div by zero
    dt = np.gradient(t, edge_order=2)
    vx = np.gradient(px, dt, edge_order=2)
    vy = np.gradient(py, dt, edge_order=2)
    return np.sqrt(np.nan_to_num(vx) ** 2 + np.nan_to_num(vy) ** 2)


def gather_keypoints(df: pd.DataFrame, idx: int):
    joints = ["head", "neck", "nose",
              "shoulder_left", "shoulder_right",
              "elbow_left", "elbow_right",
              "wrist_left", "wrist_right",
              "hand_left", "hand_right",
              "hip_left", "hip_right",
              "knee_left", "knee_right",
              "ankle_left", "ankle_right",
              "pelvis", "spine_chest", "spine_navel", "eye_left", "eye_right"]
    kp = {}
    for j in joints:
        xcol = f"{j}_x"
        ycol = f"{j}_y"
        if xcol in df.columns and ycol in df.columns:
            xv = pd.to_numeric(df.at[idx, xcol], errors="coerce")
            yv = pd.to_numeric(df.at[idx, ycol], errors="coerce")
            if not pd.isna(xv) and not pd.isna(yv):
                kp[j] = (float(xv), float(yv))
    return kp


# ---------- plotting functions ----------
def plot_trajectory_html(rel_x, rel_y, time, df, outpath):
    d = {
        "x": rel_x,
        "y": rel_y,
        "time": time,
    }
    hover_text = []
    for i in range(len(time)):
        txt = f"t={time[i]:.2f}s"
        # attach small selection of columns if present
        for c in ["will_interact", "interacting", "greeting", "shutter_action"]:
            if c in df.columns:
                txt += f"<br>{c}={df.iloc[i].get(c)}"
        hover_text.append(txt)
    d["hover"] = hover_text
    fig = px.scatter(d, x="x", y="y", color="time", color_continuous_scale="Viridis", hover_name="hover",
                     title="Trajectory (robot-centric) colored by time")
    fig.update_traces(marker=dict(size=6))
    fig.add_trace(go.Scatter(x=[0], y=[0], marker=dict(color="red", size=12, symbol="x"), name="robot_origin"))
    fig.update_layout(yaxis_scaleanchor="x", height=700, width=700)
    fig.write_html(outpath)
    print(f"Wrote interactive trajectory to {outpath}")


def plot_time_series_html(time, dist, speed, will_interact, outpath):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Distance to robot (m)", "Speed (m/s)", "Will interact"))
    fig.add_trace(go.Scatter(x=time, y=dist, mode="lines", name="distance"), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=speed, mode="lines", name="speed"), row=2, col=1)
    if will_interact is not None:
        fig.add_trace(go.Scatter(x=time, y=will_interact, mode="markers+lines", name="will_interact"), row=3, col=1)
    fig.update_layout(height=800, title_text="Time series")
    fig.write_html(outpath)
    print(f"Wrote time-series to {outpath}")


def plot_distance_speed_html(dist, speed, will_interact, outpath):
    color = will_interact if will_interact is not None else dist
    fig = px.scatter(x=dist, y=speed, color=color, labels={"x": "distance (m)", "y": "speed (m/s)"},
                     title="Distance vs Speed (color=will_interact if present)")
    fig.write_html(outpath)
    print(f"Wrote distance vs speed to {outpath}")


def animate_person_html(df, rel_x, rel_y, time, mapping, outpath, max_frames=800):
    # Build frames: each frame contains person point and keypoints (robot-centric if possible)
    n = len(df)
    step = 1
    if n > max_frames:
        step = max(1, n // max_frames)
    frames = []
    init_data = []
    # initial scatter
    init_data.append(go.Scatter(x=[rel_x[0]], y=[rel_y[0]], mode="markers", marker=dict(size=10, color="blue"), name="person"))
    init_data.append(go.Scatter(x=[0], y=[0], marker=dict(size=12, color="red", symbol="x"), name="robot"))
    # empty keypoint traces for links will be added per frame if available
    fig = go.Figure(
        data=init_data,
        layout=go.Layout(
            title="Animation (robot-centric)",
            xaxis=dict(range=[np.nanmin(rel_x) - 1.5, np.nanmax(rel_x) + 1.5], autorange=False),
            yaxis=dict(range=[np.nanmin(rel_y) - 1.5, np.nanmax(rel_y) + 1.5], autorange=False),
            updatemenus=[dict(type="buttons",
                              buttons=[dict(label="Play",
                                            method="animate",
                                            args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}])])]
        ),
        frames=[]
    )
    # simple keypoint display as markers per frame
    for i in range(0, n, step):
        pts_x = []
        pts_y = []
        kp = gather_keypoints(df, i)
        robot_x = None
        robot_y = None
        if mapping.get("robot_x") in df.columns and mapping.get("robot_y") in df.columns:
            robot_x = pd.to_numeric(df.at[i, mapping["robot_x"]], errors="coerce")
            robot_y = pd.to_numeric(df.at[i, mapping["robot_y"]], errors="coerce")
        # convert kp to robot frame if possible
        for _, (xx, yy) in kp.items():
            if robot_x is not None and not pd.isna(robot_x):
                pts_x.append(xx - robot_x)
                pts_y.append(yy - robot_y)
            else:
                pts_x.append(xx)
                pts_y.append(yy)
        frame_data = [
            go.Scatter(x=[rel_x[i]], y=[rel_y[i]], mode="markers", marker=dict(size=10, color="blue"), name="person"),
            go.Scatter(x=[0], y=[0], marker=dict(size=12, color="red", symbol="x"), name="robot")
        ]
        if pts_x:
            frame_data.append(go.Scatter(x=pts_x, y=pts_y, mode="markers", marker=dict(size=6, color="green"), name="keypoints"))
        frames.append(go.Frame(data=frame_data, name=str(i), layout=go.Layout(title_text=f"t={float(time.iloc[i]):.2f}s")))
    fig.frames = frames
    fig.write_html(outpath)
    print(f"Wrote animation HTML to {outpath} (frames={len(frames)})")


# ---------- main ----------
def main():
    p = argparse.ArgumentParser(description="Interactive Plotly visualization for robot POV CSV")
    p.add_argument("csv", help="path to csv file")
    p.add_argument("--outdir", default="viz_plotly_out", help="output directory")
    p.add_argument("--person-anchor", default=None, help="person anchor prefix (pelvis/head)")
    p.add_argument("--robot-anchor", default=None, help="robot anchor prefix (robot_base/robot_head)")
    p.add_argument("--no-animate", action="store_true", help="skip animation")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_csv(args.csv)

    mapping = choose_columns(df, person_anchor=args.person_anchor, robot_anchor=args.robot_anchor)
    if "person_x" not in mapping or "robot_x" not in mapping:
        print("Auto-detection failed: available columns (sample):")
        print(list(df.columns)[:120])
        print("Rerun with --person-anchor <prefix> and --robot-anchor <prefix> (e.g., pelvis, robot_base).")
        return

    time = ensure_time(df, mapping.get("time"))
    rel_x, rel_y = to_robot_frame(df, mapping["person_x"], mapping["person_y"], mapping["robot_x"], mapping["robot_y"])
    dist = np.sqrt(rel_x ** 2 + rel_y ** 2)
    speed = compute_speed(df, mapping["person_x"], mapping["person_y"], time)

    will_interact = None
    for col in ["will_interact", "interacting", "currently_interacting", "future_interaction"]:
        if col in df.columns:
            will_interact = pd.to_numeric(df[col], errors="coerce").to_numpy()
            break

    # Trajectory HTML
    out_traj = os.path.join(args.outdir, "trajectory_robot_frame.html")
    plot_trajectory_html(rel_x, rel_y, time.to_numpy(), df, out_traj)

    # Time-series HTML
    out_ts = os.path.join(args.outdir, "time_series.html")
    plot_time_series_html(time.to_numpy(), dist, speed, will_interact, out_ts)

    # Distance vs speed
    out_ds = os.path.join(args.outdir, "distance_vs_speed.html")
    plot_distance_speed_html(dist, speed, will_interact, out_ds)

    # Animation (optional)
    if not args.no_animate:
        try:
            out_anim = os.path.join(args.outdir, "animation.html")
            animate_person_html(df, rel_x, rel_y, time, mapping, out_anim)
        except Exception as e:
            print("Animation error (falling back):", e)
            print("You can re-run with --no-animate to skip animation.")

    print("All done. Open the generated .html files in a browser.")


if __name__ == "__main__":
    main()