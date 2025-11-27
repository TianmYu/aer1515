from pathlib import Path
import numpy as np
import cv2
import tempfile, os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import face_landmark as fl
import gaze_mlp
import data_pipeline

if __name__ == "__main__":
    # configuration for ad-hoc demo when running this file directly
    ann_path = Path('day09/p10.txt')
    image_rel = 'day09/0001.jpg'
    model_asset = Path('face_landmarker.task')
    out_dir = Path('tmp_out')
    out_dir.mkdir(exist_ok=True)

    # read annotation line for the chosen image
    lines = [l.strip() for l in ann_path.read_text().splitlines() if l.strip()]
    line = None
    for l in lines:
        if l.startswith(image_rel):
            line = l
            break
    if line is None:
        raise SystemExit(f'No annotation line found for {image_rel}')

    tokens = line.split()
    print('num tokens', len(tokens))
    for i,t in enumerate(tokens):
        print(i, t)

    # As used elsewhere in the project: face center = fields 22..24 (1-based) -> indices 21..23
    # gaze target = fields 25..27 -> indices 24..26
    fc = np.array([float(tokens[21]), float(tokens[22]), float(tokens[23])])
    gt = np.array([float(tokens[24]), float(tokens[25]), float(tokens[26])])
    vec = gt - fc
    norm = np.linalg.norm(vec)
    unit = vec / (norm + 1e-12)
    # Apply transform for MPIIGaze images so GT is in display convention
    if data_pipeline.is_mpii_gaze_image(image_rel):
        unit = data_pipeline.apply_dataset_coordinate_transform(unit, image_rel)
    print('\nParsed ground-truth:')
    print('face_center (fc):', fc)
    print('gaze_target (gt):', gt)
    print('gt - fc:', vec)
    print('unit(gt - fc):', unit)

    # Setup mediapipe face landmarker
    base_options = python.BaseOptions(model_asset_path=str(model_asset))
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # create mediapipe image and run detect
    img_path = Path(image_rel)
    if not img_path.exists():
        raise SystemExit(f'Image not found: {img_path}')
    mp_img = mp.Image.create_from_file(str(img_path))
    res = detector.detect(mp_img)

    if not hasattr(res, 'face_landmarks') or len(res.face_landmarks) == 0:
        print('No face detected')
    else:
        face_landmarks = res.face_landmarks[0]
        print('\nFound landmarks count:', len(face_landmarks))
        # print first few landmarks
        for i, lm in enumerate(face_landmarks[:10]):
            print(i, lm.x, lm.y, lm.z)

        # convert mp.Image to RGB numpy for drawing
        # mp.Image has a method to_numpy() in newer versions, but if not available, read via cv2
        try:
            img_rgb = mp_img.numpy_view()
        except Exception:
            import numpy as _np
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # compute gaze (no overlay) and draw a thin face mesh first, then draw gaze overlay
        gaze = fl.gaze_est(res, image_rgb=img_rgb, refine_pupil=False, map_to_angles=True)
        # draw thin mesh on a copy of the image
        try:
            mesh_vis = fl.draw_thin_landmarks_on_image(img_rgb, res, line_color=(10,200,10), landmark_color=(10,255,10), thickness=1, circle_radius=1)
        except Exception:
            # fallback to original image if mesh drawing fails
            mesh_vis = img_rgb.copy()
        # draw gaze overlay (ellipses + mediapipe-derived arrows) on top of the mesh
        vis = fl.draw_gaze_overlay(mesh_vis, gaze, detection_result=res, left_color=(30,200,30), right_color=(30,30,220), arrow_scale=0.12)
        print('\nGaze estimation (gaze_norm_avg_cam, gaze_yaw_pitch_deg):')
        print(gaze.get('gaze_norm_avg_cam'))
        print(gaze.get('gaze_yaw_pitch_deg'))

        # draw a ground-truth arrow from the eye midpoint using gt-fc direction
        try:
            h, w = img_rgb.shape[:2]
            # eye midpoint: average of left/right iris centers if available
            eye_pts = []
            for side in ('left', 'right'):
                if gaze.get(side) and gaze.get(side).get('iris_center'):
                    eye_pts.append(gaze[side]['iris_center'])
            if len(eye_pts) == 0:
                # fallback: use nose/mid-face landmark indices from detection_result
                flm = res.face_landmarks[0]
                # use landmark 1 (near nose bridge) as proxy
                lm = flm[1]
                eye_mid = (lm.x, lm.y)
            else:
                eye_mid = (sum([p[0] for p in eye_pts]) / len(eye_pts), sum([p[1] for p in eye_pts]) / len(eye_pts))

            # compute unit direction from gt - fc (we computed earlier)
            dir3 = unit
            # map X,Y components to image px deltas: X -> right, Y -> up (image y grows downwards)
            arrow_scale_px = 0.25 * min(h, w)
            dx = int(round(dir3[0] * arrow_scale_px))
            dy = -int(round(dir3[1] * arrow_scale_px))
            start = (int(round(eye_mid[0] * w)), int(round(eye_mid[1] * h)))
            end = (start[0] + dx, start[1] + dy)
            # draw in red
            try:
                vis = vis.copy()
                cv2.arrowedLine(vis, start, end, (0, 0, 255), 3, tipLength=0.25)
                # mark start
                cv2.circle(vis, start, 4, (0, 0, 255), -1)
                print('\nDrew ground-truth arrow from', start, 'to', end)
            except Exception as e:
                print('Failed to draw GT arrow:', e)

        except Exception as e:
            print('GT arrow drawing skipped:', e)

        # save overlay image
        out_path = out_dir / 'one_image_check.jpg'
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), vis_bgr)
        print('\nWrote overlay to', out_path)

    print('\nDone.')


def save_debug_overlay_from_annotation(line: str, detector, model=None, feat_mean=None, feat_std=None, device=None, out_path: Path = None, img_index: int = 1):
        """Given a single annotation line and a mediapipe face landmarker `detector`,
        produce and save an overlay image. If `model` is provided along with
        `feat_mean`/`feat_std` and `device`, the model prediction will be drawn
        in green; the ground-truth arrow is drawn in red.

        `out_path` should be a Path where the overlay will be written.
        """
        toks = line.split()
        if len(toks) < 27:
            raise ValueError('annotation line has insufficient tokens')
        # parse fc and gt
        fc = np.array([float(toks[21]), float(toks[22]), float(toks[23])])
        gt = np.array([float(toks[24]), float(toks[25]), float(toks[26])])
        vec = gt - fc
        nrm = np.linalg.norm(vec)
        unit = vec / (nrm if nrm > 1e-12 else 1e-12)
        # Apply transform for MPIIGaze images so GT is in display convention
        if data_pipeline.is_mpii_gaze_image(toks[0]):
            unit = data_pipeline.apply_dataset_coordinate_transform(unit, toks[0])

        img_path = Path(toks[0])
        # Resolve relative paths robustly: prefer absolute, else try module-relative,
        # then try common MPIIGazeSet prefix so annotations with that token resolve.
        if not img_path.exists():
            base_dir = Path(__file__).resolve().parent
            candidate = base_dir / toks[0]
            if candidate.exists():
                img_path = candidate
            else:
                # try MPIIGazeSet/<orig>
                candidate2 = base_dir / Path('MPIIGazeSet') / toks[0]
                if candidate2.exists():
                    img_path = candidate2
        if not img_path.exists():
            raise FileNotFoundError(f'Image not found: {img_path}')

        # Some dataset images are single-channel; MediaPipe expects 3-channel
        # images for this model. Read with OpenCV and convert to 3-channel if
        # needed, writing a small temporary file for MediaPipe to consume.
        tmp_created = False
        try:
            img_cv = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img_cv is None:
                raise FileNotFoundError(f'Could not read image: {img_path}')
            if img_cv.ndim == 2 or (img_cv.ndim == 3 and img_cv.shape[2] == 1):
                img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
                tf = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                cv2.imwrite(tf.name, img_cv_rgb)
                tf.close()
                mp_img = mp.Image.create_from_file(str(tf.name))
                tmp_created = True
            else:
                mp_img = mp.Image.create_from_file(str(img_path))
        except Exception:
            # fallback to letting MediaPipe load the file directly
            mp_img = mp.Image.create_from_file(str(img_path))
        res = detector.detect(mp_img)
        try:
            img_rgb = mp_img.numpy_view()
        except Exception:
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if not hasattr(res, 'face_landmarks') or len(res.face_landmarks) == 0:
            # save original image as-is to indicate no face
            outp = out_path or (Path('tmp_out') / f'no_face_{img_index}.jpg')
            cv2.imwrite(str(outp), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            return outp

        face_landmarks = res.face_landmarks[0]
        try:
            mesh_vis = fl.draw_thin_landmarks_on_image(img_rgb, res, line_color=(10,200,10), landmark_color=(10,255,10), thickness=1, circle_radius=1)
        except Exception:
            mesh_vis = img_rgb.copy()
        # Do not use MediaPipe-derived gaze estimates as GT; only draw mesh and use annotation GT.
        vis = mesh_vis.copy()

        # compute eye midpoint from raw landmarks (iris indices) without using gaze_est
        h, w = img_rgb.shape[:2]
        indices = fl.face_landmark_indices()
        eye_pts = []
        # left iris
        try:
            lips = indices.left_iris
            lpts = [(face_landmarks[i].x, face_landmarks[i].y) for i in lips if i < len(face_landmarks)]
            if len(lpts) > 0:
                left_center = (sum([p[0] for p in lpts]) / len(lpts), sum([p[1] for p in lpts]) / len(lpts))
                eye_pts.append(left_center)
        except Exception:
            pass
        # right iris
        try:
            rips = indices.right_iris
            rpts = [(face_landmarks[i].x, face_landmarks[i].y) for i in rips if i < len(face_landmarks)]
            if len(rpts) > 0:
                right_center = (sum([p[0] for p in rpts]) / len(rpts), sum([p[1] for p in rpts]) / len(rpts))
                eye_pts.append(right_center)
        except Exception:
            pass

        if len(eye_pts) == 0:
            flm = res.face_landmarks[0]
            lm = flm[1]
            eye_mid = (lm.x, lm.y)
        else:
            eye_mid = (sum([p[0] for p in eye_pts]) / len(eye_pts), sum([p[1] for p in eye_pts]) / len(eye_pts))

        arrow_scale_px = 0.25 * min(h, w)
        # Correct vector for display: if z < 0, flip to match yaw/pitch calculation
        display_vec = unit.copy()
        if display_vec[2] < 0:
            display_vec = -display_vec
        dx = int(round(display_vec[0] * arrow_scale_px))
        dy = -int(round(display_vec[1] * arrow_scale_px))
        start = (int(round(eye_mid[0] * w)), int(round(eye_mid[1] * h)))
        end = (start[0] + dx, start[1] + dy)
        vis = vis.copy()
        cv2.arrowedLine(vis, start, end, (0, 0, 255), 3, tipLength=0.25)
        cv2.circle(vis, start, 4, (0, 0, 255), -1)

        # if model provided, compute prediction and draw (green)
        if model is not None and feat_mean is not None and feat_std is not None and device is not None:
            try:
                lm_pts = [(float(lm.x), float(lm.y), float(lm.z)) for lm in face_landmarks]
                feat = gaze_mlp.landmarks_to_feature_vector(lm_pts)
                feat_norm = (feat - feat_mean) / feat_std
                import torch as _torch
                x_in = _torch.from_numpy(feat_norm.astype(np.float32)).unsqueeze(0).to(device)
                model.eval()
                with _torch.no_grad():
                    outp = model(x_in)
                if isinstance(outp, tuple) or (hasattr(outp, '__len__') and len(outp) == 2):
                    dir_raw_pred = outp[0]
                else:
                    dir_raw_pred = outp
                if dir_raw_pred.dim() == 1:
                    dir_raw_pred = dir_raw_pred.unsqueeze(0)
                pred_norm = _torch.sqrt(_torch.clamp((dir_raw_pred ** 2).sum(dim=1, keepdim=True), min=1e-8))
                dir_unit_pred = (dir_raw_pred / pred_norm).cpu().numpy().reshape(-1)
                
                # Model outputs in training space which IS display space (transform applied during training)
                # Correct for display: if z < 0, flip to match yaw/pitch calculation
                display_pred = dir_unit_pred.copy()
                if display_pred[2] < 0:
                    display_pred = -display_pred
                
                dxp = int(round(display_pred[0] * arrow_scale_px))
                dyp = -int(round(display_pred[1] * arrow_scale_px))
                pred_end = (start[0] + dxp, start[1] + dyp)
                cv2.arrowedLine(vis, start, pred_end, (0, 255, 0), 2, tipLength=0.2)
                cv2.circle(vis, pred_end, 3, (0, 255, 0), -1)
            except Exception:
                pass

        # compute yaw/pitch for GT and prediction (degrees)
        def vec_to_yaw_pitch_deg(v):
            # Standardize: world frame uses +x right, +y up, +z forward (away from camera).
            # If z < 0 (vector points toward camera), flip for stable angles.
            x, y, z = float(v[0]), float(v[1]), float(v[2])
            import math
            if z < 0:
                x, y, z = -x, -y, -z
            # Yaw: rotation around Y axis; positive when looking right.
            yaw = math.degrees(math.atan2(x, z))
            # Pitch: rotation around X axis; positive when looking up.
            horiz = math.sqrt(max(0.0, x * x + z * z))
            if horiz <= 1e-12:
                pitch = 0.0
            else:
                pitch = math.degrees(math.atan2(y, horiz))
            return yaw, pitch

        gt_yaw, gt_pitch = vec_to_yaw_pitch_deg(unit)
        pred_yaw = pred_pitch = None
        try:
            if model is not None and 'dir_unit_pred' in locals():
                pred_yaw, pred_pitch = vec_to_yaw_pitch_deg(dir_unit_pred)
        except Exception:
            pred_yaw = pred_pitch = None

        # overlay text lines
        lines = []
        lines.append(f'GT yaw:{gt_yaw:+.2f}\u00B0 pitch:{gt_pitch:+.2f}\u00B0')
        if pred_yaw is not None and pred_pitch is not None:
            lines.append(f'PRED yaw:{pred_yaw:+.2f}\u00B0 pitch:{pred_pitch:+.2f}\u00B0')

        # draw text with background for readability
        x0, y0 = 10, 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        # compute background rect size
        max_w = 0
        total_h = 0
        for i, ln in enumerate(lines):
            (w, h), _ = cv2.getTextSize(ln, font, font_scale, thickness)
            max_w = max(max_w, w)
            total_h += h + 8
        rect_pt1 = (x0 - 6, y0 - 16)
        rect_pt2 = (x0 + max_w + 6, y0 + total_h)
        cv2.rectangle(vis, rect_pt1, rect_pt2, (0, 0, 0), -1)
        # render each line
        yy = y0
        for ln in lines:
            cv2.putText(vis, ln, (x0, yy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            yy += int(20 * font_scale) + 6

        outp = out_path or (Path('tmp_out') / f'overlay_{img_index}.jpg')
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(outp), vis_bgr)
        return outp


def save_debug_overlay_from_image(img_path: str, detector, model=None, feat_mean=None, feat_std=None, device=None, out_path: Path = None, img_index: int = 1):
    """Run face landmark detection on an image path and save overlay with model prediction (no GT).

    Returns the saved `Path`.
    """
    img_p = Path(img_path)
    if not img_p.exists():
        raise FileNotFoundError(f'Image not found: {img_p}')

    # Prepare MediaPipe image, handling single-channel originals
    tmp_created = False
    try:
        img_cv = cv2.imread(str(img_p), cv2.IMREAD_UNCHANGED)
        if img_cv is None:
            raise FileNotFoundError(f'Could not read image: {img_p}')
        if img_cv.ndim == 2 or (img_cv.ndim == 3 and img_cv.shape[2] == 1):
            img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
            tf = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(tf.name, img_cv_rgb)
            tf.close()
            mp_img = mp.Image.create_from_file(str(tf.name))
            tmp_created = True
            img_rgb = img_cv_rgb[:, :, ::-1]
        else:
            mp_img = mp.Image.create_from_file(str(img_p))
            try:
                img_rgb = mp_img.numpy_view()
            except Exception:
                img_bgr = cv2.imread(str(img_p))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        # last-resort fallback
        mp_img = mp.Image.create_from_file(str(img_p))
        try:
            img_rgb = mp_img.numpy_view()
        except Exception:
            img_bgr = cv2.imread(str(img_p))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize to minimum dimension for consistent overlay size
    h, w = img_rgb.shape[:2]
    min_dim = 800
    if min(h, w) < min_dim:
        scale = min_dim / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # Recreate mp_img with resized image
        tf2 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        img_bgr_tmp = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tf2.name, img_bgr_tmp)
        tf2.close()
        mp_img = mp.Image.create_from_file(str(tf2.name))
        if tmp_created:
            try:
                import os
                os.unlink(tf.name)
            except:
                pass
        tmp_created = True
        tf = tf2

    res = detector.detect(mp_img)

    if not hasattr(res, 'face_landmarks') or len(res.face_landmarks) == 0:
        outp = out_path or (Path('tmp_out') / f'no_face_{img_index}.jpg')
        cv2.imwrite(str(outp), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        if tmp_created:
            try:
                os.remove(tf.name)
            except Exception:
                pass
        return outp

    face_landmarks = res.face_landmarks[0]
    try:
        mesh_vis = fl.draw_thin_landmarks_on_image(img_rgb, res, line_color=(10,200,10), landmark_color=(10,255,10), thickness=1, circle_radius=1)
    except Exception:
        mesh_vis = img_rgb.copy()
    vis = mesh_vis.copy()

    # find eye midpoint from iris indices if available
    h, w = img_rgb.shape[:2]
    indices = fl.face_landmark_indices()
    eye_pts = []
    try:
        lips = indices.left_iris
        lpts = [(face_landmarks[i].x, face_landmarks[i].y) for i in lips if i < len(face_landmarks)]
        if len(lpts) > 0:
            left_center = (sum([p[0] for p in lpts]) / len(lpts), sum([p[1] for p in lpts]) / len(lpts))
            eye_pts.append(left_center)
    except Exception:
        pass
    try:
        rips = indices.right_iris
        rpts = [(face_landmarks[i].x, face_landmarks[i].y) for i in rips if i < len(face_landmarks)]
        if len(rpts) > 0:
            right_center = (sum([p[0] for p in rpts]) / len(rpts), sum([p[1] for p in rpts]) / len(rpts))
            eye_pts.append(right_center)
    except Exception:
        pass
    if len(eye_pts) == 0:
        flm = res.face_landmarks[0]
        lm = flm[1]
        eye_mid = (lm.x, lm.y)
    else:
        eye_mid = (sum([p[0] for p in eye_pts]) / len(eye_pts), sum([p[1] for p in eye_pts]) / len(eye_pts))

    arrow_scale_px = 0.25 * min(h, w)

    # if model provided, run prediction and draw green arrow
    pred_yaw = pred_pitch = None
    if model is not None and feat_mean is not None and feat_std is not None and device is not None:
        try:
            lm_pts = [(float(lm.x), float(lm.y), float(lm.z)) for lm in face_landmarks]
            feat = gaze_mlp.landmarks_to_feature_vector(lm_pts)
            feat_norm = (feat - feat_mean) / feat_std
            import torch as _torch
            x_in = _torch.from_numpy(feat_norm.astype(np.float32)).unsqueeze(0).to(device)
            model.eval()
            with _torch.no_grad():
                outp = model(x_in)
            if isinstance(outp, tuple) or (hasattr(outp, '__len__') and len(outp) == 2):
                dir_raw_pred = outp[0]
            else:
                dir_raw_pred = outp
            if dir_raw_pred.dim() == 1:
                dir_raw_pred = dir_raw_pred.unsqueeze(0)
            pred_norm = _torch.sqrt(_torch.clamp((dir_raw_pred ** 2).sum(dim=1, keepdim=True), min=1e-8))
            dir_unit_pred = (dir_raw_pred / pred_norm).cpu().numpy().reshape(-1)
            
            # Model outputs in training space which IS display space
            # No transform needed
            
            dxp = int(round(dir_unit_pred[0] * arrow_scale_px))
            dyp = -int(round(dir_unit_pred[1] * arrow_scale_px))
            pred_end = (int(round(eye_mid[0] * w)) + dxp, int(round(eye_mid[1] * h)) + dyp)
            start = (int(round(eye_mid[0] * w)), int(round(eye_mid[1] * h)))
            # draw a higher-contrast, thicker arrow so prediction is visible on varied backgrounds
            cv2.arrowedLine(vis, start, pred_end, (0, 255, 255), 4, tipLength=0.22)
            cv2.circle(vis, pred_end, 4, (0, 255, 255), -1)
            # compute yaw/pitch
            def vec_to_yaw_pitch_deg(v):
                # Consistent with GT: +x right, +y up, +z forward.
                x, y, z = float(v[0]), float(v[1]), float(v[2])
                import math
                if z < 0:
                    x, y, z = -x, -y, -z
                yaw = math.degrees(math.atan2(x, z))
                horiz = math.sqrt(max(0.0, x * x + z * z))
                if horiz <= 1e-12:
                    pitch = 0.0
                else:
                    pitch = math.degrees(math.atan2(y, horiz))
                return yaw, pitch
            pred_yaw, pred_pitch = vec_to_yaw_pitch_deg(dir_unit_pred)
        except Exception as e:
            print(f'Prediction failed for {img_p.name}:', e)

    # overlay predicted yaw/pitch text
    lines = []
    if pred_yaw is not None and pred_pitch is not None:
        lines.append(f'PRED yaw:{pred_yaw:+.2f}\u00B0 pitch:{pred_pitch:+.2f}\u00B0')
        # make text background high-contrast (bright yellow text on dark box)
        # we render as white on black rectangle already; add a small label marker
        # prepend a marker dot to help identify prediction lines quickly
        lines[0] = '\u25CF ' + lines[0]
    # draw text
    x0, y0 = 10, 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    max_w = 0
    total_h = 0
    for i, ln in enumerate(lines):
        (w_txt, h_txt), _ = cv2.getTextSize(ln, font, font_scale, thickness)
        max_w = max(max_w, w_txt)
        total_h += h_txt + 8
    if len(lines) > 0:
        rect_pt1 = (x0 - 6, y0 - 16)
        rect_pt2 = (x0 + max_w + 6, y0 + total_h)
        cv2.rectangle(vis, rect_pt1, rect_pt2, (0, 0, 0), -1)
        yy = y0
        for ln in lines:
            cv2.putText(vis, ln, (x0, yy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            yy += int(20 * font_scale) + 6

    outp = out_path or (Path('tmp_out') / f'coco_overlay_{img_index}.jpg')
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(outp), vis_bgr)
    if tmp_created:
        try:
            os.remove(tf.name)
        except Exception:
            pass
    return outp
