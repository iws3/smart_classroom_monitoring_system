# Track 4: Dashboard, Alerting & Visual Overlays

## 🚀 What I Have Done

- **Audio Alert Module:** Built a non-blocking `Alarm` class using `playsound3` with an integrated cooldown mechanism to trigger sound alerts without stalling the video frame processing loop.
- **Visual Overlay System:** Implemented helper functions (`get_pose_bbox_px` and `draw_flag_overlay`) to calculate pixel-space bounding boxes from normalized pose landmarks and render target overlays over flagged students.
- **Spotlight Background Effect:** Created a `spotlight_bbox` function to dim the surrounding classroom feed while keeping the flagged student's region fully visible.
- **Event Logging:** Integrated structured CSV logging to record alert events (timestamps, student IDs, and suspicion metrics) for post-session review.
- **OpenCV Interactive Control:** Added interactive key bindings (`a` to trigger/flag, `r` to reset, `q` to quit) to manually test the alarm, bounding box, and spotlight states.

---

## I did not understand

- **Visual Overlays & Math:** I did not understand how to convert normalized MediaPipe coordinates into exact pixel positions to draw red bounding boxes and crosses around students accurately.
