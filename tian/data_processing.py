import cv2 as cv
from ultralytics import YOLO
import numpy as np
import time
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import torch.utils.data as data
import torch

# keypoint mapping for v11
keypoint_names = [
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Ear",
    "Right Ear",
    "Left Shoulder", # 5
    "Right Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Wrist", 
    "Right Wrist", 
    "Left Hip", # 11 
    "Right Hip",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle"
]

keypoint_names_training = [
    "nose",
    "eye_left",
    "eye_right",
    "ear_left",
    "ear_right",
    "shoulder_left",  # 5
    "shoulder_right",
    "elbow_left",
    "elbow_right",
    "wrist_left",
    "wrist_right",
    "hip_left",  # 11
    "hip_right",
    "knee_left",
    "knee_right",
    "ankle_left",
    "ankle_right"
]

# extract yolo pose and face direction from video. do it together to prevent multiple video loads
def load_video_get_features(video_path, show = True):
    # get the yolo model
    model = YOLO("yolo11l-pose.pt") 
    # model sizes: n, s, m, l, x
    yolo_full = model(video_path) #, stream = True)
    # produce a np array of size [n, 17, d]
    ex_data = yolo_full[0].keypoints.data.detach().cpu().numpy()
    n_people, pts, dim = ex_data.shape

    out = np.zeros((len(yolo_full), pts, dim))
    out_boxes = np.zeros((len(yolo_full), 4))
    # for i in range(0, len(yolo_full)):
    #     frame_result = yolo_full[i].keypoints.data.detach().cpu().numpy()
    #     box = yolo_full[i].boxes.xyxy.detach().cpu().numpy()
    #     out[i, :, :] = frame_result[0]
    #     out_boxes[i,:] = box[0]

    # # if we want to extract a specific person
    # out = out[:, :, :]
    # out_boxes = out_boxes[:,:]

    return [out], [out_boxes], yolo_full

# convert yolo extracted results into model compatible form
# TODO: need to split into sub series if it jumps from person to person
def process_yolo_output(yolo_results, k, input_freq = 30, target_freq = 5):
    n, pts, dim = yolo_results.shape

    k_i = np.linalg.inv(k)

    # convert frequency (downsample) 
    freq_ratio = input_freq / target_freq
    indices = [0]
    ratio_track = freq_ratio
    while ratio_track < n:
        indices += [ratio_track]
        ratio_track += freq_ratio
    indices = np.array(indices)
    indices = np.rint(indices).astype(int) # TODO: do some linear interpolation
    yolo_proc = yolo_results[indices].copy()

    # project from pixels into normalized frame and slice out the "confidence" dim
    yolo_proc = yolo_proc.reshape((-1, dim)).T
    yolo_proc[2, :] = 1

    yolo_proc = k_i @ yolo_proc
    yolo_proc = (yolo_proc.T).reshape(len(indices), pts, dim)
    yolo_proc = yolo_proc[:, :, 0:2]

    # compress all features into 1 dimension and shift order of dims
    seq_len = len(yolo_proc)
    yolo_proc_out = yolo_proc.reshape(seq_len, -1).transpose(1, 0)

    return [yolo_proc_out], [indices]

# draw our ingestion and prediction results
# all results objects need to be an iterable with the same number of entries as video frames
def draw_features(video_path, yolo_results, gaze_results = None, yolo_boxes = None, pred_results = None, pred_indices = None, 
                  fps_target = 30, max_frame = None):
    cap = cv.VideoCapture(video_path)
    show_name = "frame"
    frame_n = 0
    pred_idx_tracker = 0

    try:
        while cap.isOpened():
            t1 = time.perf_counter()
            ret, frame = cap.read()

            if not ret:
                print("no frame returned")
                break

            # draw the yolo results over the frame
            yolo_frame = frame.copy()
            yolo_result = yolo_results[frame_n]
            yolo_frame = draw_yolo(yolo_frame, yolo_result)

            # draw the intent prediction box
            if yolo_boxes is not None:
                yolo_box = yolo_boxes[frame_n]
                # index tracker, allows for scaling between frame number and index to frame num mapping of pred indices array
                if pred_idx_tracker < len(pred_indices) - 1:
                    if frame_n > pred_indices[pred_idx_tracker]:
                        pred_idx_tracker += 1

                pred_result = pred_results[pred_idx_tracker]
                yolo_frame = draw_box(yolo_frame, yolo_box, pred_result)

            # concat and imshow
            show_frames = [frame, yolo_frame]
            show_frame = np.concatenate(show_frames, axis = 1)
            show_frame = cv.putText(show_frame, f"{frame_n-frame_n%5}", (5, 15),
                            cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv.imshow(show_name, show_frame)
            if cv.waitKey(1) == ord('q'):
                break

            frame_n += 1
            if max_frame is not None and frame_n > max_frame:
                break

            t2 = time.perf_counter()
            sleep_time = 1/fps_target - (t2-t1)
            if sleep_time > 0:
                time.sleep(sleep_time)
            t3 = time.perf_counter()

            cv.waitKey(1)
    finally:
        cv.destroyWindow(show_name)

def draw_yolo(frame, yolo_result):
    # yolo results is an n x 17 x 3 matrix of n people, 17 body parts, and 3 dim of x, y, confidence
    data = yolo_result
    pts, dim = data.shape
    
    pose_img = frame
    # draw skeleton
    for j in range(0, 5):
        center = np.rint(data[j, 0:2]).astype(int)
        pose_img = cv.circle(pose_img, center, radius=3, color=(0,255,0), thickness=-1)

    for j in range(5, 11):
        center = np.rint(data[j, 0:2]).astype(int)
        pose_img = cv.circle(pose_img, center, radius=3, color=(0,0,255), thickness=-1)

    for j in range(11, pts):
        center = np.rint(data[j, 0:2]).astype(int)
        pose_img = cv.circle(pose_img, center, radius=3, color=(255,0,0), thickness=-1)

    # draw lines
    index_pairs = [
        (3, 1, (0,255,0)), # face
        (1, 0), 
        (0, 2),
        (2, 4),
        (9, 7, (0,0,255)), # left wrist > shoulder > rigth wrist
        (7, 5),
        (5, 6),
        (6, 8),
        (8, 10),
        (5, 11), # shoulders to hips
        (6, 12),
        (11, 13, (255,0,0)), # hips to knees
        (13, 15),
        (12, 14), 
        (14, 16),
    ]

    for j in range(0, len(index_pairs)):
        index_pair = index_pairs[j]
        if len(index_pair) == 3:
            color = index_pair[2]

        start_pt = np.rint(data[index_pair[0], 0:2]).astype(int)
        end_pt = np.rint(data[index_pair[1], 0:2]).astype(int)

        pose_img = cv.line(pose_img, start_pt, end_pt, color, thickness = 2)

    return pose_img

def draw_box(frame, box_coords, pred, pred_thresh = 0.5):
    if pred >= pred_thresh:
        color = (0, 255, 0)
        label = "intent"
    else:
        color = (255, 0, 0)
        label = "no intent"

    x1, y1, x2, y2 = np.rint(box_coords).astype(int)
    frame = cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if y1 > 15: 
        y1_text = y1-5
    else:
        y1_text = y1+15

    frame = cv.putText(frame, label, (x1 + 5, y1_text),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame

# given a list of dpt classes, plot all the points in a real time animation 
def draw_points_matplot(dpt_list, use_array = False, only_2d = False):
    # assemble x, y, z arrays

    x_arr_list = []
    y_arr_list = []
    z_arr_list = []

    label_arr = None
    gaze_orig = None
    gaze_vect = None

    # if clause for processing different input types
    if use_array:
        # formatted as a np array n x npts x d
        n, npts, d = dpt_list.shape

        x_arr = dpt_list[:, :, 0].transpose(1, 0)
        y_arr = dpt_list[:, :, 1].transpose(1, 0)
        z_arr = np.zeros_like(y_arr)

        label_arr = np.zeros(n)
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

    if only_2d:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection='3d')

    x = x_arr[:,0]
    y = y_arr[:,0]
    z = z_arr[:,0]

    if only_2d:
        sc = ax.scatter(x, y, c='tomato')
    else:
        sc = ax.scatter(x, y, z, c='tomato')

    print(x_arr.shape)
    ax.set_xlim(np.min(x_arr) - 0.5, np.max(x_arr) + 0.5)
    ax.set_ylim(np.min(y_arr) - 0.5, np.max(y_arr) + 0.5)

    if not only_2d:
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

        if only_2d:
            sc.set_offsets(np.column_stack((new_x, new_y)))
        else:
            sc._offsets3d = (new_x, new_y, new_z)
        return (sc, quiv)
    
    update_func = partial(update, quiv = None) # [quiv])

    ani = FuncAnimation(fig, update_func, frames=x_arr.shape[1], interval=50)

    return ani
    # plt.show()
    # plt.close()

# animate a specific data series given np array of dim x n
# dim is 34, which is xy for each point in the order of keypoint names
def plot_validate(data_arr):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    data_len, dim = data_arr.shape
    x_idx = [i for i in range(0, dim, 2)]
    y_idx = [i for i in range(1, dim, 2)]
    x = data_arr[0, x_idx]
    y = data_arr[0, y_idx]

    colors = ["red"] * 5 + ["blue"] * 6 + ["green"] * (len(x_idx) - 11)

    sc = ax.scatter(x, y, c=colors)

    ax.set_xlim(np.min(data_arr[:, x_idx]) - 0.5, np.max(data_arr[:, x_idx]) + 0.5)
    ax.set_ylim(np.min(data_arr[:, y_idx]) - 0.5, np.max(data_arr[:, y_idx]) + 0.5)

    def update(frame):
        new_x = data_arr[frame, x_idx]
        new_y = data_arr[frame, y_idx]
        
        ax.set_title(f"frame: {frame}")

        sc.set_offsets(np.column_stack((new_x, new_y)))
        return sc,
    
    update_func = partial(update) 
    ani = FuncAnimation(fig, update_func, frames=data_len, interval=50)
    return ani

# convert list of data point set objects to list of np arrays with data and labels
# also drops x axis if required
def convert_dptset_list(dptset_list):
    data_out_list = []
    label_out_list = []

    for dptset in dptset_list:
        data_list = []
        for key in keypoint_names_training:
            data_list += [dptset.data_points[key][1:,:]]
        data_arr = np.concatenate(data_list, axis = 0)

        data_out_list += [data_arr]
        label_out_list +=[dptset.labels]

    return data_out_list, label_out_list


# dpt set list needs to be a list of np arrays of dimension d x n
# windows the sequences, and returns it in np array format for model ingestion
def create_sequences(dpt_arr_list, seq_length, stride, label_arr_list = None, time_offset_secs = 0):
    # note the data we had is 5hz
    time_offset_i = time_offset_secs * 5

    X, y = [], []
    for j, dpt_array in enumerate(dpt_arr_list):
        data_len = dpt_array.shape[1]
        if label_arr_list is not None:
            label_array = label_arr_list[j]
        
        # also filters out any sequences shorter than seq len + time offset
        for i in range(0, data_len - seq_length - time_offset_i, stride): 
            X.append(dpt_array[:, i:i+seq_length])

            if label_arr_list is not None:
                if True in label_array[i + time_offset_i:i+ time_offset_i + seq_length]:
                    y.append([True])
                else:
                    y.append([False])
    
    X = np.array(X)
    X = np.transpose(X, (0, 2, 1))

    if label_arr_list is not None:
        y = np.array(y)
    else: 
        y = None
    return X, y

# to 0-1 for now, need to extend to saving scaling factors
# assumes format N x seq len x d
def scale_seq(seq):
    N, seq_len, d = seq.shape

    for i in range(0, d):
        seq[:,:,i] = (seq[:,:,i] - seq[:,:,i].min())/(seq[:,:,i].max() - seq[:,:,i].min())
    return seq

def create_dataloaders(X, y, train_ratio = 0.6, batch_size = 32):
    dataset = data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    train_dataset, val_dataset = data.random_split(dataset, [train_ratio, 1-train_ratio])
    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = data.DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
    return train_loader, val_loader

def get_rates(y_true, y_pred):
    y_true_in = y_true.detach().numpy().flatten()
    y_pred_in = y_pred.detach().numpy().flatten()

    y_pred_in[y_pred_in >= 0.5] = 1
    y_pred_in[y_pred_in < 0.5] = 0

    y_true_in = y_true_in.astype(bool)
    y_pred_in = y_pred_in.astype(bool)

    tp = np.sum(y_true_in & y_pred_in)
    tn = np.sum(~y_true_in & ~y_pred_in)

    fp = np.sum(~y_true_in & y_pred_in)
    fn = np.sum(y_true_in & ~y_pred_in)

    return np.array([tp, tn, fp, fn])

def get_precis_recall(rates):
    tp = rates[0]
    tn = rates[1]
    fp = rates[2]
    fn = rates[3]

    if tp == 0:
        precis, recall = 0
        print("warning, no true positives")
    else:
        precis = tp / (tp + fp)
        recall = tp / (tp + fn)
    return precis, recall

def plot_confusion_matrix(arr):
    """
    Expects arr = [TP, TN, FP, FN]
    Plots confusion matrix using matplotlib.
    """
    TP, TN, FP, FN = arr
    
    matrix = np.array([
        [TP, FN],
        [FP, TN]
    ])

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="Oranges")

    # Labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted Positive", "Predicted Negative"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True Positive", "True Negative"])

    # Show values inside squares
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="black", fontsize=12)

    # Titles
    ax.set_title("LSTM Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()
    plt.show()

def add_dict_list(target_dict, key, item):
    if key in target_dict:
        target_dict[key] += [item]
    else:
        target_dict[key] = [item]
    return target_dict