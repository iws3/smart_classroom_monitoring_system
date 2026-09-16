# Gesture Recognition

## 1. Hand Gesture Identifier

- I created a hand gesture identifier using the **gesture_recognizer.task** pretrained MediaPipe model.
- The system takes input from a webcam and detects hand gestures in real time.
- It was able to identify gestures such as **open palm, thumbs up, thumbs down, pointing up, pointing down, and victory**.
- I learned that MediaPipe first detects the hand landmarks and then uses them to classify the hand into a specific gesture.

## 2. Face Landmark Identifier

- I created a face landmark identifier using the **MediaPipe Face Landmarker**.
- The system detects a face from the webcam and maps out the facial structure using landmarks.
- I learned that the Face Landmarker can detect **478 facial landmarks**, which represent different parts of the face such as the eyes, nose, mouth, cheeks, and jaw.
- I also learned that these landmarks can be connected together to form a **3D face mesh**.

## 3. Face Mesh

- I used the detected face landmarks to draw the facial mesh.
- The mesh is created by connecting different landmarks together to represent the structure of the face.
- I learned that MediaPipe provides predefined connections between the landmarks, which can be used to draw the mesh instead of manually deciding which points should be connected.

## 4. Blendshapes

- I explored the **blendshape** output provided by the Face Landmarker.
- I learned that blendshapes provide scores that describe different facial expressions or movements, such as **eye blinking, smiling, eyebrow movement, and jaw opening**.
- The scores indicate how strongly a particular facial movement is being detected.
- I am still trying to understand exactly how MediaPipe calculates these scores and how they should be interpreted when building our own facial behavior rules.

## 5. Facial Transformation Matrix

- I explored the **facial transformation matrix** returned by the Face Landmarker.
- I learned that it contains information about the face's **3D rotation and translation**.
- I used the rotation information to calculate the **head pose**, particularly the yaw angle.
- This can help us determine whether a person is looking or turning their head left or right.
- I still need to understand exactly what each value in the 4 × 4 matrix represents and how the rotation values are converted into yaw, pitch, and roll.

## Issues Faced

- I don't understand how the **blendshape scores** work.

- I don't understand how the **facial transformation matrix** is deduced.

- I don't understand how the **head pose is calculated** from the transformation matrix.
