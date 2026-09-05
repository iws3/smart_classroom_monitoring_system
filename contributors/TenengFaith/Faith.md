# My Computer Vision Journey: Progress & Milestones

I recently started exploring computer vision and real-time processing. this is a breakdown of what I've covered and learnt so far:

---

## 1. MediaPipe Foundations
* Discovered what **MediaPipe** is and how its pipeline architecture differs from standard computer vision approaches.
* Got a clear grasp on how it optimizes real-time processing.

## 2. Frame Extraction & Processing
* Learned what a frame actually is in video streams.
* Extracted and processed frames from a live video feed in real time.
* Computed dynamic frame resolutions and applied frame-by-frame transformations.

## 3. Hand Landmark Detection
* Built on frame processing to dive into **MediaPipe Landmarks**.
* Focused specifically on hand landmarks using the `hand_landmarker.task` model.
* Successfully detected and tracked key hand landmarks live via webcam.

## 4. Real-Time Gesture Recognition
* Combined my understanding of frames and landmarks to move into gesture recognition.
* Integrated a pre-trained `gesture_recognizer` model to analyze hand landmarks on each frame and identify gestures on the fly.

---

### Tech Stack & Tools
* **Library / Framework:** MediaPipe, OpenCV
* **Models:** `hand_landmarker.task`, `gesture_recognizer.task`
* **Source:** Live Webcam Feed