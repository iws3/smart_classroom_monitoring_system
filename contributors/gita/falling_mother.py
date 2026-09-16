import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
# GET THE BODY PARTS WE WANT
L_SHOULDER, R_SHOULDER=11, 12
L_HIP, R_HIP=23, 24
L_ANKLE, R_ANKLE=27, 28
NOSE=0
MIN_VISIBILITY=0.5

def landmarks_visible(lm, indices):
    return all(lm[i].visibility>=MIN_VISIBILITY for i in indices)


def body_bbox_aspect_ratio(lm, h, w):
    visible_points=[(p.x*w, p.y*h) for p in lm if p.visibility>=MIN_VISIBILITY]
    if len(visible_points) < 4:
        return None

    xs=[p[0] for p in visible_points]
    ys=[p[1] for p in visible_points]
    # print(f"x values: {xs}")
    # print(f"y values: {ys}")
    box_w=max(xs)-min(xs)
    box_h=max(ys)-min(ys)
    if box_h < 1e-3:
        return None
    return box_w/box_h 






POSE_CONNECTIONS=vision.PoseLandmarksConnections.POSE_LANDMARKS
JOINT_STYLE = vision.drawing_utils.DrawingSpec(
    color=(0, 255, 255), thickness=2, circle_radius=4
)
BONE_STYLE = vision.drawing_utils.DrawingSpec(
    color=(255, 0, 0), thickness=3
)

def main():
    source=sys.argv[1] if len(sys.argv)> 1 else 0
    base_options=python.BaseOptions(model_asset_path="pose_landmarker_full.task")
    options=vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )

    detector=vision.PoseLandmarker.create_from_options(options)
    cap=cv2.VideoCapture(source)
    fps=cap.get(cv2.CAP_PROP_FPS) or 30
    frame_duration_ms=1000/fps
    frame_timestamp_ms=0


    STATE="NORMAL"
    fall_timer_start=None
    hip_y_history=[]
    frame_count=0

    while cap.isOpened():
        success, frame=cap.read()
        if not success:
            break

        frame_count+=1

        h, w=frame.shape[:2]
        rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image=mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms+=frame_duration_ms
        result=detector.detect_for_video(mp_image, int(frame_timestamp_ms))

        if result.pose_landmarks:
            lm=result.pose_landmarks[0]
            vision.drawing_utils.draw_landmarks(
                frame, lm, POSE_CONNECTIONS,
                landmark_drawing_spec=JOINT_STYLE,
                connection_drawing_spec=BONE_STYLE

            )
            torsor_visibility=landmarks_visible(lm, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP])
            aspect_ratio=body_bbox_aspect_ratio(lm, w, h)

            # if aspect_ratio < 1.5:

            if frame_count%40==0:
                if torsor_visibility:
                    print("All parts are visible......")
                if aspect_ratio < 1.3:
                    print("standing up")
                else:
                    print("lying down")

        cv2.imshow("Fall Detecion", frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
        # break
    cap.release()
    cv2.destroyAllWindows()


main()
    


