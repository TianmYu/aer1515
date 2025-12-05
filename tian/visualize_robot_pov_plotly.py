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
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

YOLO_KEYPTS = [
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Ear",
    "Right Ear",
    "Left Shoulder",
    "Right Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Wrist",
    "Right Wrist",
    "Left Hip",
    "Right Hip",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle"
]

class DataPointSet:
    def __init__(self, times, labels = None, gaze = None):
        # data points is a dict of name: 3xN numpy array of points xyz
        # sin and cos angles are 1xN numpy arrays

        self.times = times
        self.labels = labels
        self.data_points = {}
        self.gaze = gaze

    def parse_datapoints(self, dpt_df):
        # parses pd df with <name>_x/y/z into the data point format shown above
        base_names = [col[:-2] for col in dpt_df.columns if col[-2:] == "_x"]
        for name in base_names:
            points_df = dpt_df[[name+"_x", name+"_y", name+"_z"]]
            points_np = points_df.to_numpy().T # dim 3xN

            self.data_points[name] = points_np

    # assume yaw angle is fixed for duration of the data series and we only have translation in yaw
    def transform_datapoints(self, translation_arr, yaw_angle):
        # translate data points by translation array, and yaw by yaw_arr

        yaw_rot_matrix = np.array([[np.cos(yaw_angle), -np.sin(yaw_angle), 0],
                                   [np.sin(yaw_angle), np.cos(yaw_angle), 0],
                                   [0, 0, 1]])
        
        for key, val in self.data_points.items():
            self.data_points[key] = yaw_rot_matrix.T @ (self.data_points[key] - translation_arr)
    
    # divide by depth to get normalized coordinates (camera equivalent), depth assumed to be in x axis 
    def normalize_camera(self):
        for key, val in self.data_points.items():
            self.data_points[key] = self.data_points[key] / self.data_points[key][0,:]

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
    df = df.replace({"True": True, "False": False})
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
            if verbose:
                print(f"Expanded '{col}' -> {[f'{col}_{a}' for a in ['x','y','z'][:max_len]]}")
    # try cast obvious numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    return df

def extract_columns(df: pd.DataFrame) -> Dict[str, str]:
    timestamp_colname = "shutter_timestamp"
    label_cols = ["interacting"]

    # take the robot pos as robot_head (note, not robot_head_0/1/etc). also consider cos and sin
    rob_cols_base = ["robot_base"]
    dims = ["x", "y", "z", "yaw"]
    angles = [None]
    # angles = [None, "cos", "sin"]
    cols = []

    for col in rob_cols_base:
        for dim in dims:
            for angle in angles:
                new_col = col + "_" + dim
                if angle is not None:
                    new_col = new_col + "_" + angle
                cols += [new_col]
    
    robot_df = df[[timestamp_colname] + cols].copy()
    robot_df[timestamp_colname] = robot_df[timestamp_colname] - robot_df[timestamp_colname].iloc[0] # shift timestamp to start at 0

    robot_dpts = DataPointSet(robot_df[timestamp_colname].to_numpy())
    robot_dpts.parse_datapoints(robot_df)

    trans_arr = robot_dpts.data_points[rob_cols_base[0]]
    yaw_df = robot_df[rob_cols_base[0] + "_yaw"]
    if max(yaw_df) - min(yaw_df) > 0.1:
        print("warning, non negligible yaw difference detected")
    yaw = yaw_df.iloc[0]

    robot_dpts.transform_datapoints(trans_arr, yaw)

    # get yolo columns from the yolo list above. dont include any sin/cos cols
    cols = []
    for col in df.columns:
        if "sin" in col or "cos" in col:
            continue

        for yolo_name in YOLO_KEYPTS:
            yolo_name_words = yolo_name.lower().split(" ")
            both_in = True
            for word in yolo_name_words:
                if word not in col:
                    both_in = False
            if both_in:
                cols += [col]

    human_df = df[[timestamp_colname] + label_cols + cols].copy()
    human_df[timestamp_colname] = human_df[timestamp_colname] - human_df[timestamp_colname].iloc[0] # shift timestamp to start at 0

    # extract gaze vectors - this is probably actually robot gaze
    # gaze_cols = ["gaze_pos_x", "gaze_pos_y", "gaze_pos_z"]
    gaze_df = None # df[gaze_cols].copy()

    human_dpts = DataPointSet(human_df[timestamp_colname].to_numpy(), labels = human_df[label_cols].to_numpy(), gaze = None)
    human_dpts.parse_datapoints(human_df)
    human_dpts.transform_datapoints(trans_arr, yaw)
    human_dpts.normalize_camera()

    # plt.plot(range(0, len(df.index)), human_df[label_cols].to_numpy())
    # plt.show()
    return robot_df, human_df, gaze_df, robot_dpts, human_dpts

# ---------- plotting functions ----------
 
# given a list of dpt classes, plot all the points in a real time animation 
def draw_points_matplot(dpt_list, use_array = False):
    # assemble x, y, z arrays

    x_arr_list = []
    y_arr_list = []
    z_arr_list = []

    label_arr = None
    gaze_orig = None
    gaze_vect = None

    # if clause for processing different input types
    if use_array:
        # formatted 
        pass
    else:
        for dpt_obj in dpt_list:
            dpt_dict = dpt_obj.data_points
            for key, val in dpt_dict.items():
                x_arr_list += [val[0:1,:]]
                y_arr_list += [val[1:2,:]]
                z_arr_list += [val[2:3,:]]
            if dpt_obj.labels is not None:
                label_arr = dpt_obj.labels

            if "nose" in dpt_dict:
                gaze_orig = dpt_dict["nose"]
                gaze_vect = dpt_obj.gaze

    x_arr = np.concatenate(x_arr_list, axis = 0)
    y_arr = np.concatenate(y_arr_list, axis = 0)
    z_arr = np.concatenate(z_arr_list, axis = 0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = x_arr[:,0]
    y = y_arr[:,0]
    z = z_arr[:,0]
    sc = ax.scatter(x, y, z, c='tomato')

    ax.set_xlim(np.min(x_arr) - 0.5, np.max(x_arr) + 0.5)
    ax.set_ylim(np.min(y_arr) - 0.5, np.max(y_arr) + 0.5)
    ax.set_zlim(np.min(z_arr) - 0.5, np.max(z_arr) + 0.5)
    # quiv = ax.quiver(gaze_orig[0, 0], gaze_orig[1, 0], gaze_orig[2, 0], gaze_vect[0, 0], gaze_vect[1, 0], gaze_vect[2, 0], length=1.0, normalize=True)

    def update(frame, quiv):
        # quiv[0].remove()
        new_x = x_arr[:,frame]
        new_y = y_arr[:,frame]
        new_z = z_arr[:,frame]

        # for gaze
        # quiv[0] = ax.quiver(gaze_orig[0, frame], gaze_orig[1, frame], gaze_orig[2, frame], gaze_vect[0, frame], gaze_vect[1, frame], gaze_vect[2, frame], length=1.0, normalize=True)
        
        ax.set_title(f"frame: {frame}")
        if label_arr[frame]:
            sc.set_color('blue')
        else:
            sc.set_color("tomato")

        sc._offsets3d = (new_x, new_y, new_z)
        return (sc, quiv)
    
    update_func = partial(update, quiv = None) # [quiv])

    ani = FuncAnimation(fig, update_func, frames=x_arr.shape[1], interval=50)
    plt.show()
    plt.close()

# ---------- main ----------
def main():
    dir = r"..\datasets\pose-extracted\lobby_1_processed_csv_files\all_files_csv"
    outdir = r"..\datasets\processed\\"
    verbose = False
    os.makedirs(outdir, exist_ok=True)

    robot_trajs = []           
    human_trajs = []

    for i, file in enumerate(os.listdir(dir)):
        if file[-4:] == ".csv":
            filepath = os.path.join(dir, file)
            df = load_csv(filepath, verbose = verbose)

            df_dict = {}
            for value, df in df.groupby('pid'):
                df_dict[value] = df.copy().reset_index()

            for pid, df_person in df_dict.items():
                robot_df, human_df, gaze_df, robot_dpts, human_dpts = extract_columns(df_person)
                draw_points_matplot([robot_dpts, human_dpts])

                robot_trajs += [robot_dpts]
                human_trajs += [human_dpts]

            print(f"loaded {i}/{len(os.listdir(dir))}")
            # if i > 10:
                 #break
    
    out_path = os.path.join(outdir, "preproc_out_2.pkl")
    with open(out_path, "wb") as f:  # "wb" = write binary
        pickle.dump((robot_trajs, human_trajs), f)
    print("data pickled")

if __name__ == "__main__":
    main()