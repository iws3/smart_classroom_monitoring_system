I have gone through gesture recognition using mediapipe and this is what i have done so far:

## 1. hand gesture identifier:
* I created a hand gesture identifier using the **gesture_recognizer.task** pretrained model which was able to identify four hand gestures (open palm, thumbs up&down, pointing up&down, victory) directly from a webcam feed

## 2. face gesture recognizer:
* I created a face landmark identifier to identify facial landmarks and map out the facial structure


### issues faced
* I have no idea how the blendshape score and exactly how it works
* I am confused on what the facial_recognition_matrix stores and how it is used to deduce the headpose
* I do not understand how the landmarks are being linked to form the facemesh

