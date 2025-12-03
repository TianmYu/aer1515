import cv2 as cv
from ultralytics import YOLO
import numpy as np
import time
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

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

# extract yolo pose and face direction from video. do it together to prevent multiple video loads
def load_video_get_features(video_path, show = True):
    # get the yolo model
    model = YOLO("yolo11l-pose.pt") 
    # model sizes: n, s, m, l, x
    yolo_results = model(video_path) #, stream = True)
    # produce a np array of size [n, 17, d]
    ex_data = yolo_results[0].keypoints.data.detach().cpu().numpy()
    n_people, pts, dim = ex_data.shape

    out = np.zeros((len(yolo_results), n_people, pts, dim))
    for i in range(0, len(yolo_results)):
        frame_result = yolo_results[i].keypoints.data.detach().cpu().numpy()
        out[i, :, :, :] = frame_result

    # if we want to extract a specific person
    out = out[:, 0, :, :]

    return [out]

# convert yolo extracted results into model compatible form
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

    # project into normalized frame and slice out the confidence dim
    yolo_proc = yolo_proc.reshape((-1, dim)).T
    yolo_proc[2, :] = 1
    print(yolo_proc.shape)
    print(yolo_proc)
    yolo_proc = k_i @ yolo_proc
    yolo_proc = (yolo_proc.T).reshape(len(indices), pts, dim)
    yolo_proc_out = yolo_proc[:, :, 0:2]

    return yolo_proc_out

# draw our ingestion and prediction results
# all results objects need to be an iterable with the same number of entries as video frames
def draw_features(video_path, yolo_results, gaze_results = None, prediction_results = None):
    cap = cv.VideoCapture(video_path)
    show_name = "frame"
    frame_n = 0
    goal_framerate = 30

    while cap.isOpened():
        t1 = time.time()
        ret, frame = cap.read()

        if not ret:
            print("no frame returned")
            break

        # draw the yolo results over the frame
        yolo_frame = frame.copy()
        yolo_result = yolo_results[frame_n]
        yolo_frame = draw_yolo(yolo_frame, yolo_result)

        show_frames = [frame, yolo_frame]
        show_frame = np.concatenate(show_frames, axis = 1)

        # draw gaze results if we have them

        # concat and imshow
        cv.imshow(show_name, show_frame)
        if cv.waitKey(1) == ord('q'):
            break

        frame_n += 1

        t2 = time.time()
        if t2 - t1 < 1/goal_framerate:
            time.sleep((1/goal_framerate) - (t2-t1))
        else:
            print("warning: running to slow for desired framerate")
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