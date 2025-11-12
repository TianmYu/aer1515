from ultralytics import YOLO
import cv2

# keypoint mapping for v11
keypoint_names = [
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

def test():
    # Load the pretrained model
    model = YOLO("yolo11n-pose.pt") 

    # Train the model
    results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
    print(results)

    return model

def draw_pose(img, results, show = True):
    pose_img = img.copy()
    for result in results:
        keypoints = result.keypoints.xy.detach().cpu().numpy().reshape((-1,2))
        data = result.keypoints.data.detach().cpu().numpy().reshape((-1,3))
        for i in range(0, len(keypoints)):
            center = (int(keypoints[i][0]), int(keypoints[i][1]))
            color = (0, 0, 255)
            pose_img = cv2.circle(pose_img, center, radius=5, color=color, thickness=-1)
            pose_img = cv2.putText(pose_img, keypoint_names[i], (center[0] + 5, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            pose_img = cv2.putText(pose_img, f"{data[i,2]:.2f}", (center[0] + 5, center[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    if show:
        cv2.imshow("img with joint keypoints", pose_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return pose_img

