"""
Fall Detector
-------------
Uses MediaPipe Pose to detect when a person falls, based on:
  - torso angle vs. vertical (upright ~0-15deg, fallen ~60-90deg)
  - vertical velocity of the hip midpoint (fast drop = fall, not a slow crouch)
  - a "stayed down" confirm timer, to reject false positives like
    bending over or sitting down quickly

State machine:
  NORMAL -> SUSPECTED_FALL -> FALL_CONFIRMED
  (SUSPECTED_FALL reverts to NORMAL if the person gets back up quickly)

Requires: pose_landmarker_lite.task in the same folder
(download: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task)

Run:
    python fall_detector.py            # uses webcam
    python fall_detector.py video.mp4  # uses a video file
"""

import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_heavy.task"

JOINT_STYLE = vision.drawing_utils.DrawingSpec(
    color=(0, 255, 255), thickness=2, circle_radius=4
)
BONE_STYLE = vision.drawing_utils.DrawingSpec(
    color=(255, 0, 0), thickness=3
)
POSE_CONNECTIONS = vision.PoseLandmarksConnections.POSE_LANDMARKS

def torso_angle_from_vertical(shoulder_mid, hip_mid):
    """Angle between the hip->shoulder vector and straight-up, in degrees.
    0deg = standing upright. 90deg = torso horizontal (bent all the way over).
    ~180deg = shoulder below the hip on screen (fully folded/collapsed).

    IMPORTANT: do NOT abs() dx/dy before this calculation - that throws away
    the up/down direction and makes a collapsed body look "upright" again.
    """
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]  # image y grows downward, so "up" is -y
    return abs(np.degrees(np.arctan2(dx, -dy)))


L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28
NOSE = 0

# --- tunable thresholds ---
CONFIRM_SECONDS = 2.0     # how long the person must stay down to confirm a fall
RECOVER_ANGLE = 30        # torso angle below this = standing again
FALL_ANGLE = 60           # torso angle above this = suspected fall
VELOCITY_THRESHOLD = 15   # pixel drop over 5 frames to count as a fast fall
MIN_VISIBILITY = 0.5      # ignore a landmark if the model isn't confident it's actually visible
ASPECT_RATIO_FALL = 1.3   # bbox width/height above this = lying down, even if shoulders aren't visible
HEAD_DROP_THRESHOLD = 0.25  # (hip.y - nose.y) / (ankle.y - nose.y) below this = head has dropped to hip level


def landmarks_visible(lm, indices):
    """True only if every requested landmark has a visibility score above threshold."""
    return all(lm[i].visibility >= MIN_VISIBILITY for i in indices)


def head_drop_ratio(lm, w, h):
    """(hip.y - nose.y) / (ankle.y - nose.y), in pixel coordinates.
    ~0.5+ when standing (nose well above hip). Drops toward 0, then negative,
    as the head folds down to hip height or below - this works even when a
    frontal camera view makes the shoulder-hip angle barely change (foreshortening),
    which is exactly the case a forward bow/collapse viewed face-on produces."""
    needed = [NOSE, L_HIP, R_HIP, L_ANKLE, R_ANKLE]
    if not landmarks_visible(lm, needed):
        return None
    nose_y = lm[NOSE].y * h
    hip_y = (lm[L_HIP].y + lm[R_HIP].y) / 2 * h
    ankle_y = (lm[L_ANKLE].y + lm[R_ANKLE].y) / 2 * h
    denom = ankle_y - nose_y
    if abs(denom) < 1e-3:
        return None
    return (hip_y - nose_y) / denom


def body_bbox_aspect_ratio(lm, w, h):
    """Width/height of the bounding box over all confidently-visible landmarks.
    A lying-down body is wide and short (>1); a standing body is narrow and tall (<1).
    Useful when shoulders/head are out of frame and torso_angle can't be trusted."""
    visible_points = [(p.x * w, p.y * h) for p in lm if p.visibility >= MIN_VISIBILITY]
    if len(visible_points) < 4:
        return None
    xs = [p[0] for p in visible_points]
    ys = [p[1] for p in visible_points]
    box_w = max(xs) - min(xs)
    box_h = max(ys) - min(ys)
    if box_h < 1e-3:
        return None
    return box_w / box_h


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture("falling.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_duration_ms = 1000 / fps
    frame_timestamp_ms = 0

    STATE = "NORMAL"
    fall_timer_start = None
    hip_y_history = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms += frame_duration_ms
        result = detector.detect_for_video(mp_image, int(frame_timestamp_ms))

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            vision.drawing_utils.draw_landmarks(
                frame, lm, POSE_CONNECTIONS,
                landmark_drawing_spec=JOINT_STYLE,
                connection_drawing_spec=BONE_STYLE,
            )

            torso_visible = landmarks_visible(lm, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP])
            aspect_ratio = body_bbox_aspect_ratio(lm, w, h)
            head_ratio = head_drop_ratio(lm, w, h)

            torso_angle = None
            if torso_visible:
                shoulder_mid = [(lm[L_SHOULDER].x + lm[R_SHOULDER].x) / 2 * w,
                                 (lm[L_SHOULDER].y + lm[R_SHOULDER].y) / 2 * h]
                hip_mid = [(lm[L_HIP].x + lm[R_HIP].x) / 2 * w,
                           (lm[L_HIP].y + lm[R_HIP].y) / 2 * h]
                torso_angle = torso_angle_from_vertical(shoulder_mid, hip_mid)

                # track hip Y over the last 5 frames to estimate vertical velocity
                hip_y_history.append(hip_mid[1])
                if len(hip_y_history) > 5:
                    hip_y_history.pop(0)
            vertical_velocity = (hip_y_history[-1] - hip_y_history[0]) if len(hip_y_history) == 5 else 0

            # a fall/collapse is signalled by ANY of:
            #   - large torso angle (works for sideways falls, where the shoulder
            #     visibly swings away from vertical in the 2D projection)
            #   - wide/flat bounding box (works even with shoulders out of frame)
            #   - head dropped to hip height (works for a frontal-view forward
            #     fold/collapse, where the torso ANGLE barely changes on screen
            #     due to foreshortening, but the head height is unambiguous)
            angle_signal = torso_visible and torso_angle > FALL_ANGLE
            shape_signal = aspect_ratio is not None and aspect_ratio > ASPECT_RATIO_FALL
            head_signal = head_ratio is not None and head_ratio < HEAD_DROP_THRESHOLD
            is_down = angle_signal or shape_signal or head_signal

            recovered = (not angle_signal or torso_visible and torso_angle < RECOVER_ANGLE) and \
                        (aspect_ratio is None or aspect_ratio < (ASPECT_RATIO_FALL * 0.7)) and \
                        (head_ratio is None or head_ratio > (HEAD_DROP_THRESHOLD * 1.6))

            if STATE == "NORMAL":
                # skip the velocity requirement for signals that don't rely on
                # motion (head/shape) - a slow collapse should still be caught
                if is_down and (vertical_velocity > VELOCITY_THRESHOLD or shape_signal or head_signal or not torso_visible):
                    STATE = "SUSPECTED_FALL"
                    fall_timer_start = time.time()

            elif STATE == "SUSPECTED_FALL":
                if recovered:
                    STATE = "NORMAL"  # got back up quickly -> false alarm
                elif time.time() - fall_timer_start > CONFIRM_SECONDS:
                    STATE = "FALL_CONFIRMED"

            elif STATE == "FALL_CONFIRMED":
                cv2.putText(frame, "FALL DETECTED - ALERT", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                # TODO: trigger a real alert here, e.g. SMS/push via Twilio
                if recovered:
                    STATE = "NORMAL"

            angle_str = f"{int(torso_angle)}deg" if torso_angle is not None else "N/A"
            ratio_str = f"{aspect_ratio:.2f}" if aspect_ratio is not None else "N/A"
            head_str = f"{head_ratio:.2f}" if head_ratio is not None else "N/A"
            cv2.putText(frame, f"State: {STATE}",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Torso:{angle_str} AR:{ratio_str} Head:{head_str}",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()