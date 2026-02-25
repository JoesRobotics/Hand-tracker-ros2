# mediapipe_hands_viser

Ready-to-run ROS 2 (Python) package that:
- Captures from a camera using OpenCV
- Runs MediaPipe Hands to detect left/right hands and 3D world landmarks
- Publishes `/hands/left` and `/hands/right` as `geometry_msgs/msg/PoseArray`
- Visualizes 3D landmarks with a Viser web viewer at http://localhost:8080

## Dependencies

ROS 2 Humble (or newer) with Python:

- `rclpy`
- `geometry_msgs`

Python packages (install into the same environment as your ROS 2 tools):

```bash
pip install mediapipe opencv-python viser
```

## Build

Assuming you have a workspace `~/ros2_ws`:

```bash
cd ~/ros2_ws/src
# Place this folder here: mediapipe_hands_viser/
cd ..
colcon build --packages-select mediapipe_hands_viser
source install/setup.bash
```

## Run

```bash
ros2 launch mediapipe_hands_viser mediapipe_hands_viser.launch.py
```

Then open your browser at:

- http://localhost:8080

To inspect the PoseArray topics:

```bash
ros2 topic echo /hands/left
ros2 topic echo /hands/right
```

## Pinch Controls

In the Viser GUI panel ("Mediapipe Hands Controller") you get:

- **Publish to ROS** checkbox – turn PoseArray publishing on/off.
- **Pinch close threshold** slider – normalized threshold (0..1) for deciding when a pinch is considered CLOSED.
- **Set pinch OPEN pose** button – click while your thumb/index are fully open to capture the "open" reference.
- **Set pinch CLOSED pose** button – click while thumb/index are pinched to capture the "closed" reference.
- Live readouts:
  - Pinch distance (meters) between thumb tip and index tip.
  - Pinch angle (deg) between wrist->thumb and wrist->index vectors.
  - Normalized pinch progress in [0, 1].
  - Pinch state: NO HAND / UNCALIBRATED / OPEN / CLOSED.

The 3D viewer shows both **joints (spheres)** and **links (line segments)** for each detected hand, giving you a full hand skeleton visualization.

## Live Wrist Pose & Pinch State Topics

This node also publishes:

- `/hands/left/wrist_pose`  (`geometry_msgs/PoseStamped`)
- `/hands/right/wrist_pose` (`geometry_msgs/PoseStamped`)

Each PoseStamped contains the 3D position of the wrist landmark in MediaPipe world coordinates
(frame_ids: `hand_left` and `hand_right`, orientation = identity).

Pinch-related topics (based on whichever hand is currently used for pinch, preferring right):

- `/hand/pinch/progress` (`std_msgs/Float32`)
  - `-1.0` if uncalibrated or no hand.
  - Otherwise in `[0, 1]` where 0 = calibrated OPEN pose, 1 = calibrated CLOSED pose.
- `/hand/pinch/closed` (`std_msgs/Bool`)
  - `true` when pinch is considered CLOSED given the current threshold.
- `/hand/pinch/hand` (`std_msgs/String`)
  - `"right"` or `"left"` for the active pinch hand, or `""` if none.

You can subscribe to these for live control signals, e.g.:

```bash
ros2 topic echo /hands/right/wrist_pose
ros2 topic echo /hand/pinch/progress
ros2 topic echo /hand/pinch/closed
```
