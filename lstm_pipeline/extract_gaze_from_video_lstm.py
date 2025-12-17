"""Extract gaze angles from video frames.

Returns a numpy array with gaze angles for each frame.

Usage:
    from extract_gaze_from_video import get_gaze_angles
    angles = get_gaze_angles('path/to/video.avi')
"""
import numpy as np
import cv2
import torch
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import gaze_mlp


def draw_gaze_arrow(
    frame: np.ndarray,
    gaze_vector: np.ndarray,
    face_landmarks,
    arrow_length: int = 80,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    draw_facemesh: bool = False,
) -> np.ndarray:
    """Draw gaze arrow on video frame.
    
    Args:
        frame: BGR image as numpy array
        gaze_vector: 3D gaze vector (x, y, z)
        face_landmarks: MediaPipe face landmarks
        arrow_length: Length of the arrow in pixels
        color: Arrow color in BGR
        thickness: Arrow line thickness
        draw_facemesh: Whether to draw facial mesh overlay
    
    Returns:
        Frame with gaze arrow drawn
    """
    height, width = frame.shape[:2]
    frame_copy = frame.copy()
    
    # Draw face mesh if requested
    if draw_facemesh:
        connections = [
            # Face contour
            (10, 338), (338, 297), (297, 332), (332, 284), (284, 251),
            (251, 389), (389, 356), (356, 454), (454, 323), (323, 361),
            (361, 288), (288, 397), (397, 365), (365, 379), (379, 378),
            (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
            (176, 149), (149, 150), (150, 136), (136, 172), (172, 58),
            (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
            (162, 21), (21, 54), (54, 103), (103, 67), (67, 69),
            (69, 108), (108, 10),
            # Left eyebrow
            (46, 53), (53, 52), (52, 65), (65, 55),
            # Right eyebrow
            (276, 283), (283, 282), (282, 295), (295, 285),
            # Left eye
            (33, 7), (7, 163), (163, 144), (144, 145), (145, 153),
            (153, 154), (154, 155), (155, 133), (133, 33),
            # Right eye
            (263, 249), (249, 390), (390, 373), (373, 374), (374, 380),
            (380, 381), (381, 382), (382, 362), (362, 263),
            # Nose
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
            (6, 8), (8, 9), (9, 10),
            # Lips outer
            (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267),
            (267, 269), (269, 270), (270, 409),
            # Additional face features
            (76, 77), (77, 78), (78, 79), (79, 80),
            (306, 307), (307, 308), (308, 309), (309, 310),
        ]
        
        for idx1, idx2 in connections:
            if idx1 < len(face_landmarks) and idx2 < len(face_landmarks):
                pt1 = face_landmarks[idx1]
                pt2 = face_landmarks[idx2]
                x1 = int(pt1.x * width)
                y1 = int(pt1.y * height)
                x2 = int(pt2.x * width)
                y2 = int(pt2.y * height)
                cv2.line(frame_copy, (x1, y1), (x2, y2), (100, 200, 100), 1)
        
        # Draw all landmark points as small dots
        for lm in face_landmarks:
            x = int(lm.x * width)
            y = int(lm.y * height)
            cv2.circle(frame_copy, (x, y), 1, (0, 150, 0), -1)
    
    # Get face center (nose tip landmark 1)
    nose_tip = face_landmarks[1]
    center_x = int(nose_tip.x * width)
    center_y = int(nose_tip.y * height)
    
    # Use z-flipped vector for drawing (same as angle calculation)
    display_vec = gaze_vector.copy()
    if display_vec[2] < 0:
        display_vec = -display_vec
    
    # Draw arrow
    end_x = int(center_x + display_vec[0] * arrow_length)
    end_y = int(center_y - display_vec[1] * arrow_length)  # Flip Y for image coords
    
    cv2.arrowedLine(frame_copy, (center_x, center_y), (end_x, end_y),
                   color, thickness, tipLength=0.4)
    
    # Calculate and draw angles
    import math
    x, y, z = float(display_vec[0]), float(display_vec[1]), float(display_vec[2])
    if z < 0:
        x, y, z = -x, -y, -z
    yaw = math.degrees(math.atan2(x, z))
    horiz = math.sqrt(max(0.0, x * x + z * z))
    pitch = 0.0 if horiz <= 1e-12 else math.degrees(math.atan2(y, horiz))
    
    text = f"Yaw: {yaw:.1f} Pitch: {pitch:.1f}"
    cv2.putText(frame_copy, text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return frame_copy


def get_gaze_angles(video_path: str) -> np.ndarray:
    """Extract gaze angles from video.
    
    Assumes model files are in face_and_pose/tmp_out/:
    - gaze_mlp_balanced_6040_400ep_v2.pt
    - gaze_norm_stats.npz
    - face_landmarker.task in face_and_pose/
    
    Args:
        video_path: Path to input video file
    
    Returns:
        numpy array of shape (num_frames, 3) with columns [yaw_deg, pitch_deg, valid]
        where valid is 1 if face detected, 0 otherwise
    """
    # Setup paths
    script_dir = Path(__file__).parent
    model_path = script_dir / 'tmp_out' / 'gaze_mlp_balanced_6040_400ep_v2.pt'
    norm_stats_path = script_dir / 'tmp_out' / 'gaze_norm_stats.npz'
    face_model_path = script_dir / 'face_landmarker.task'
    
    device = 'cpu'
    min_detection_confidence = 0.5
    
    # Setup detector
    base_options = python.BaseOptions(model_asset_path=str(face_model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=min_detection_confidence,
        min_face_presence_confidence=min_detection_confidence,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    
    # Load normalization stats
    if norm_stats_path.exists():
        stats = np.load(norm_stats_path)
        feat_mean, feat_std = stats['mean'], stats['std']
    else:
        print(f"Warning: normalization stats not found at {norm_stats_path}")
        feat_mean = np.zeros(1434, dtype=np.float32)
        feat_std = np.ones(1434, dtype=np.float32)
    
    # Load model
    device_obj = torch.device(device)
    model = gaze_mlp.SimpleMLP(in_dim=1434, hidden=128, out_dim=3, dropout=0.15)
    model.load_state_dict(torch.load(str(model_path), map_location=device_obj))
    model.to(device_obj)
    model.eval()
    
    # Open video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    print(f"Processing video: {total_frames} frames @ {fps:.1f} fps")
    print("debug")
    
    # Initialize results array
    results = np.zeros((total_frames, 3), dtype=np.float32)
    
    frame_num = 0
    processed = 0
    
    draw_out = []

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                # Extract landmarks
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                res = detector.detect(mp_image)
                
                if hasattr(res, 'face_landmarks') and len(res.face_landmarks) > 0:
                    # Get landmarks
                    face_landmarks = res.face_landmarks[0]
                    lm_pts = [(float(lm.x), float(lm.y), float(lm.z)) for lm in face_landmarks]
                    
                    # Generate feature vector
                    feat = gaze_mlp.landmarks_to_feature_vector(lm_pts)
                    feat_norm = (feat - feat_mean) / feat_std
                    
                    # Predict gaze
                    x_in = torch.from_numpy(feat_norm.astype(np.float32)).unsqueeze(0).to(device_obj)
                    with torch.no_grad():
                        output = model(x_in)
                    
                    if isinstance(output, tuple) or (hasattr(output, '__len__') and len(output) == 2):
                        gaze_pred = output[0]
                    else:
                        gaze_pred = output
                    
                    if gaze_pred.dim() == 1:
                        gaze_pred = gaze_pred.unsqueeze(0)
                    
                    # Normalize to unit vector
                    norm = torch.sqrt(torch.clamp((gaze_pred ** 2).sum(dim=1, keepdim=True), min=1e-8))
                    gaze_unit = (gaze_pred / norm).cpu().numpy().flatten()
                    
                    # Calculate angles
                    import math
                    x, y, z = float(gaze_unit[0]), float(gaze_unit[1]), float(gaze_unit[2])
                    if z < 0:
                        x, y, z = -x, -y, -z
                    yaw = math.degrees(math.atan2(x, z))
                    horiz = math.sqrt(max(0.0, x * x + z * z))
                    pitch = 0.0 if horiz <= 1e-12 else math.degrees(math.atan2(y, horiz))
                    
                    draw_out += [[np.array([x, y, z]), face_landmarks, yaw, pitch]]

                    results[frame_num, 0] = yaw
                    results[frame_num, 1] = pitch
                    results[frame_num, 2] = 1  # Valid
                    processed += 1
                else:
                    draw_out += [None]
                    results[frame_num, 2] = 0  # Invalid (no face)
                    
            except Exception as e:
                draw_out += [None]
                results[frame_num, 2] = 0  # Invalid (error)
            
            frame_num += 1
            if frame_num % 100 == 0:
                print(f"Processed {frame_num}/{total_frames} frames...")
    
    finally:
        video.release()
    
    print(f"Done: {processed}/{total_frames} frames with face detected ({100*processed/total_frames:.1f}%)")
    
    return results, draw_out # array of pairs [vector, face_landmarks]


# if __name__ == '__main__':
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Extract gaze angles from video')
#     parser.add_argument('--video', required=True, help='Path to input video file')
#     parser.add_argument('--output', default=None, help='Path to output .npy file (optional)')
    
#     args = parser.parse_args()
    
#     angles = get_gaze_angles(video_path=args.video)
    
#     if args.output:
#         output_path = Path(args.output)
#         output_path.parent.mkdir(parents=True, exist_ok=True)
#         np.save(output_path, angles)
#         print(f"Saved angles array to {args.output}")
#     else:
#         print(f"\nArray shape: {angles.shape}")
#         print(f"First 5 frames (yaw, pitch, valid):")
#         print(angles[:5])
