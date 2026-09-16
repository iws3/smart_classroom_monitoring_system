# Track 2: Vision & Detection Implementation

## What I Have Done
* Set up the core detection pipeline using MediaPipe’s Vision Tasks API (`PoseLandmarker` and `FaceLandmarker`) to extract classroom vision metrics.
* Configured frame-by-frame processing for local video streams with dynamic timestamp handling to meet `VIDEO` running mode requirements.
* Built dynamic path handling so video assets and `.task` model files load reliably regardless of the terminal execution directory.
* Added an OpenCV preview window with realtime bounding box overlays to visually verify detections.

## What I Have Applied
* **MediaPipe Tasks API**: Built multi-person detection targeting up to 10 poses and faces per frame, capturing 3D keypoints and facial blendshapes.
* **OpenCV**: Handled video stream reading, color space conversions ($BGR \rightarrow RGB$), pixel scaling for bounding boxes, and frame rendering.
* **Data Structuring**: Standardized model outputs into a decoupled dictionary format (bounding boxes, keypoint coordinates, blendshape scores) ready for consumption by Track 3 (Tracking & Rule Engine).

## Repository Link
https://github.com/TenengFaith/Classroom-monitoring-and-alarm-system