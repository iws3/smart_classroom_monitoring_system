"""
apps/face_mesh_viewer_app.py

The bare face-mesh visualization script, rebuilt from reusable pieces.
No yaw, no blendshapes read, no alerting — just detect and draw, matching
the original monolithic script exactly.
"""

import cv2

from config import AppConfig
from video_source import VideoSource
from frame_utils import bgr_to_rgb, to_mp_image, FrameTimestamper
from drawing import draw_face_mesh
from detectors.face_landmarker_service import FaceLandmarkerService


class FaceMeshViewerApp:
    def __init__(self, config: AppConfig):
        self._video = VideoSource(config.video_source)
        self._detector = FaceLandmarkerService(config.face)
        self._timestamper = FrameTimestamper(config.frame_timestamp_step_ms)

    def run(self) -> None:
        while self._video.is_opened():
            success, frame = self._video.read()
            if not success:
                continue

            result = self._detect(frame)
            self._render(frame, result)

            cv2.imshow("Face Landmarker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self._shutdown()

    def _detect(self, frame):
        rgb_frame = bgr_to_rgb(frame)
        mp_image = to_mp_image(rgb_frame)
        timestamp_ms = self._timestamper.next()
        return self._detector.detect(mp_image, timestamp_ms)

    def _render(self, frame, result) -> None:
        if result.face_landmarks:
            draw_face_mesh(frame, result.face_landmarks)

    def _shutdown(self) -> None:
        self._video.release()
        self._detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    FaceMeshViewerApp(AppConfig()).run()