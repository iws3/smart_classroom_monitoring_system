frames --> landmarks --> handtracking --> gesture_recognization

1.frames 
this are still images captured in a video feed or livestream. 
since open cv aids the capture from camer it does that in the BGR format but mediapipe expects it as RGB so what do we do? we convert it
color_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
now it process it by calculating or running a certain number of FPS could be 30, 60....

2.landmarks
this is a meaningfull keypoint that a model detects on a structure
2.1 Hand Landmarker ✋
The Hand Landmarker detects a hand and returns 21 landmarks.
Important examples:
0  → Wrist
4  → Thumb tip
8  → Index fingertip
12 → Middle fingertip
16 → Ring fingertip
20 → Pinky fingertip
The 21 points describe the hand's structure and position.
Because landmarks are numbers, we can perform geometry:
Count fingers
Detect a pinch
Measure distances
Compare finger positions
Build our own gesture rules
For example, a raised non-thumb finger can be detected by comparing its fingertip's y coordinate with its PIP joint.

3.gesture recognition
landmarks describe where the hand is. A gesture gives those points meaning.
The Gesture Recognizer works roughly like this:
RGB Frame to
Hand Landmark Model to
21 Hand Landmarks to
Gesture Classifier to
Gesture + Confidence to
Built-in gesture categories include:
Closed_Fist
Open_Palm
Pointing_Up
Thumb_Down
Thumb_Up

4.Debouncing Alerts 
Never rely on just one frame for an important decision.
A detection can be wrong for a single frame. Debouncing requires the condition to remain true for several consecutive frames.
Frame 1  → True
Frame 2  → True
Frame 3  → True
...
Frame 10 → True Trigger
This helps reduce false alarms.
5. Common Pitfalls ⚠️
BGR vs RGB — convert OpenCV frames before sending them to MediaPipe.
Normalized vs pixel coordinates — multiply normalized coordinates by frame dimensions when drawing.
Empty results — always check whether landmarks were detected before indexing them.
Running modes — use the method that matches the selected mode.
FPS mismatch — a slow model can cause delayed processing.