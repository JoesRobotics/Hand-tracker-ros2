#!/usr/bin/env python3
import math
from typing import Optional, Dict

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from std_msgs.msg import Float32, Bool, String

import viser


mp_hands = mp.solutions.hands

# MediaPipe hand landmark connections for drawing the skeleton.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
]

# MediaPipe landmark indices for thumb/index pinch metrics
THUMB_TIP_IDX = 4
INDEX_TIP_IDX = 8
WRIST_IDX = 0


class MediapipeHandsViserRos(Node):
    def __init__(self):
        super().__init__('mediapipe_hands_viser_ros')

        # ---------------- Parameters ----------------
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('publish_pose_array', True)

        camera_index = self.get_parameter('camera_index').get_parameter_value().integer_value
        self.publish_pose_array = self.get_parameter('publish_pose_array').get_parameter_value().bool_value

        # ---------------- ROS 2 Publishers ----------------
        qos = 10
        # Full hand PoseArray (21 joints) per hand
        self.left_pub = self.create_publisher(PoseArray, '/hands/left', qos)
        self.right_pub = self.create_publisher(PoseArray, '/hands/right', qos)

        # Wrist pose per hand (PoseStamped)
        self.left_wrist_pub = self.create_publisher(PoseStamped, '/hands/left/wrist_pose', qos)
        self.right_wrist_pub = self.create_publisher(PoseStamped, '/hands/right/wrist_pose', qos)

        # Pinch metrics (normalized progress + closed/open state + which hand)
        self.pinch_progress_pub = self.create_publisher(Float32, '/hand/pinch/progress', qos)
        self.pinch_closed_pub = self.create_publisher(Bool, '/hand/pinch/closed', qos)
        self.pinch_hand_pub = self.create_publisher(String, '/hand/pinch/hand', qos)

        # ---------------- Camera ----------------
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.get_logger().error(f'Could not open camera index {camera_index}')
            raise RuntimeError('Camera open failed')

        # ---------------- MediaPipe Hands ----------------
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ---------------- Viser Server ----------------
        self.server = viser.ViserServer()  # default: http://localhost:8080
        self.server.scene.set_up_direction('+z')
        self.server.gui.set_panel_label('Mediapipe Hands Controller')

        # Joints as spheres (21 per hand for MediaPipe)
        self.left_spheres = [
            self.server.scene.add_icosphere(
                name=f'/left/joint/{i}',
                radius=0.01,
                color=(0, 255, 0),
                position=(0.0, 0.0, 0.0),
            )
            for i in range(21)
        ]
        self.right_spheres = [
            self.server.scene.add_icosphere(
                name=f'/right/joint/{i}',
                radius=0.01,
                color=(255, 0, 0),
                position=(0.0, 0.0, 0.0),
            )
            for i in range(21)
        ]

        # Links (bones) as line segments, batched per hand
        num_links = len(HAND_CONNECTIONS)
        self.left_link_points = np.zeros((num_links, 2, 3), dtype=float)
        self.right_link_points = np.zeros((num_links, 2, 3), dtype=float)

        self.left_links = self.server.scene.add_line_segments(
            '/left/links',
            points=self.left_link_points,
            colors=(0, 200, 0),
            line_width=2.0,
        )
        self.right_links = self.server.scene.add_line_segments(
            '/right/links',
            points=self.right_link_points,
            colors=(200, 0, 0),
            line_width=2.0,
        )

        for h in self.left_spheres + self.right_spheres:
            h.visible = False
        self.left_links.visible = False
        self.right_links.visible = False

        # ---------------- Pinch Detection State ----------------
        # We'll treat the right hand as primary for pinch; if absent, fall back to left.
        self.pinch_open_ref: Optional[Dict[str, float]] = None
        self.pinch_closed_ref: Optional[Dict[str, float]] = None
        self.last_pinch_metrics: Optional[Dict[str, float]] = None
        self.last_pinch_hand: Optional[str] = None  # 'left' or 'right'

        # ---------------- GUI Controls ----------------
        gui = self.server.gui

        self.enable_publishing_checkbox = gui.add_checkbox(
            'Publish to ROS',
            self.publish_pose_array,
            hint='Toggle publishing of PoseArray topics /hands/left and /hands/right.',
        )

        self.pinch_threshold_slider = gui.add_slider(
            'Pinch close threshold',
            0.0,
            1.0,
            0.01,
            0.5,
            hint='When normalized pinch >= threshold, state is CLOSED.',
        )

        self.pinch_distance_text = gui.add_text(
            'Pinch distance (m)',
            'distance: -',
            hint='3D distance between thumb tip and index tip.',
        )
        self.pinch_angle_text = gui.add_text(
            'Pinch angle (deg)',
            'angle: -',
            hint='Angle between wrist->thumb and wrist->index vectors.',
        )
        self.pinch_progress_text = gui.add_text(
            'Pinch progress',
            'progress: -',
            hint='0 = fully open ref, 1 = fully closed ref.',
        )
        self.pinch_state_text = gui.add_text(
            'Pinch state',
            'state: UNKNOWN',
            hint='OPEN / CLOSED based on threshold and calibrated poses.',
        )

        self.pinch_open_button = gui.add_button(
            'Set pinch OPEN pose',
            hint='Click while your thumb and index are in the fully open pose.',
        )
        self.pinch_closed_button = gui.add_button(
            'Set pinch CLOSED pose',
            hint='Click while your thumb and index are in the fully closed (pinched) pose.',
        )

        @self.pinch_open_button.on_click
        def _(_event):
            self._capture_pinch_reference(which='open')

        @self.pinch_closed_button.on_click
        def _(_event):
            self._capture_pinch_reference(which='closed')

        # Timer ~30 Hz
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        self.get_logger().info('Mediapipe+Viser+ROS2 hand node started')
        self.get_logger().info('Open browser at http://localhost:8080')

    # ---------------- Main Loop ----------------
    def timer_callback(self):
        ok, frame_bgr = self.cap.read()
        if not ok:
            self.get_logger().warning('Failed to read camera frame')
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        # If no hands: publish empty arrays, hide in viz, and publish pinch default
        if not results.multi_hand_world_landmarks:
            self._publish_empty('left')
            self._publish_empty('right')
            self._set_visible(self.left_spheres, self.left_links, False)
            self._set_visible(self.right_spheres, self.right_links, False)
            self._update_pinch_gui_no_hand()
            self._publish_pinch_state_none()
            return

        handedness_labels = []
        if results.multi_handedness:
            for h in results.multi_handedness:
                handedness_labels.append(h.classification[0].label)  # 'Left' or 'Right'

        left_world = None
        right_world = None

        for idx, hand_world in enumerate(results.multi_hand_world_landmarks):
            label = handedness_labels[idx] if idx < len(handedness_labels) else 'Unknown'
            label_low = label.lower()
            if label_low.startswith('left'):
                left_world = hand_world
            elif label_low.startswith('right'):
                right_world = hand_world

        do_publish = self.publish_pose_array and bool(self.enable_publishing_checkbox.value)

        # Left hand
        if left_world is not None:
            left_msg = self._landmarks_to_pose_array(left_world, frame_id='hand_left')
            if do_publish:
                self.left_pub.publish(left_msg)
                self._publish_wrist_pose('left', left_world)
            self._update_viser(self.left_spheres, self.left_links, self.left_link_points, left_world)
        else:
            self._publish_empty('left')
            self._set_visible(self.left_spheres, self.left_links, False)

        # Right hand
        if right_world is not None:
            right_msg = self._landmarks_to_pose_array(right_world, frame_id='hand_right')
            if do_publish:
                self.right_pub.publish(right_msg)
                self._publish_wrist_pose('right', right_world)
            self._update_viser(self.right_spheres, self.right_links, self.right_link_points, right_world)
        else:
            self._publish_empty('right')
            self._set_visible(self.right_spheres, self.right_links, False)

        # Pinch metrics: prefer right hand if available, else left
        pinch_hand = None
        pinch_hand_name = None
        if right_world is not None:
            pinch_hand = right_world
            pinch_hand_name = 'right'
        elif left_world is not None:
            pinch_hand = left_world
            pinch_hand_name = 'left'

        if pinch_hand is not None:
            self._update_pinch_gui_from_hand(pinch_hand, pinch_hand_name)
        else:
            self._update_pinch_gui_no_hand()
            self._publish_pinch_state_none()

    # ---------------- Helpers: ROS / visualization ----------------
    def _landmarks_to_pose_array(self, hand_world_landmarks, frame_id: str) -> PoseArray:
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        for lm in hand_world_landmarks.landmark:
            pose = Pose()
            pose.position.x = lm.x
            pose.position.y = lm.y
            pose.position.z = lm.z

            pose.orientation.w = 1.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0

            msg.poses.append(pose)

        return msg

    def _publish_wrist_pose(self, which: str, hand_world_landmarks):
        # Publish PoseStamped for the wrist joint of the specified hand.
        lms = hand_world_landmarks.landmark
        if WRIST_IDX >= len(lms):
            return

        wrist = lms[WRIST_IDX]

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f'hand_{which}'

        msg.pose.position.x = wrist.x
        msg.pose.position.y = wrist.y
        msg.pose.position.z = wrist.z

        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0

        if which == 'left':
            self.left_wrist_pub.publish(msg)
        else:
            self.right_wrist_pub.publish(msg)

    def _update_viser(self, spheres, links_handle, link_points_array, hand_world_landmarks):
        # Update joints (spheres)
        for handle, lm in zip(spheres, hand_world_landmarks.landmark):
            handle.position = (lm.x, lm.y, lm.z)
            handle.visible = True

        # Update links (bones) as batched line segments
        for k, (i, j) in enumerate(HAND_CONNECTIONS):
            a = hand_world_landmarks.landmark[i]
            b = hand_world_landmarks.landmark[j]
            link_points_array[k, 0, :] = (a.x, a.y, a.z)
            link_points_array[k, 1, :] = (b.x, b.y, b.z)

        links_handle.points = link_points_array
        links_handle.visible = True

    def _set_visible(self, spheres, links_handle, visible: bool):
        for h in spheres:
            h.visible = visible
        links_handle.visible = visible

    def _publish_empty(self, which: str):
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'hand_left' if which == 'left' else 'hand_right'

        if which == 'left':
            self.left_pub.publish(msg)
        else:
            self.right_pub.publish(msg)

    # ---------------- Helpers: Pinch metrics & GUI ----------------
    def _compute_pinch_metrics(self, hand_world_landmarks) -> Optional[Dict[str, float]]:
        # Return pinch distance (meters) and angle (deg) based on thumb & index tips.
        lms = hand_world_landmarks.landmark

        try:
            thumb = lms[THUMB_TIP_IDX]
            index = lms[INDEX_TIP_IDX]
            wrist = lms[WRIST_IDX]
        except IndexError:
            return None

        # Distance between thumb tip and index tip
        dx = thumb.x - index.x
        dy = thumb.y - index.y
        dz = thumb.z - index.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Angle between wrist->thumb and wrist->index
        v1 = (
            thumb.x - wrist.x,
            thumb.y - wrist.y,
            thumb.z - wrist.z,
        )
        v2 = (
            index.x - wrist.x,
            index.y - wrist.y,
            index.z - wrist.z,
        )

        def vec_norm(v):
            return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

        n1 = vec_norm(v1)
        n2 = vec_norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            angle_deg = 0.0
        else:
            dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
            cosang = max(min(dot / (n1 * n2), 1.0), -1.0)
            angle_rad = math.acos(cosang)
            angle_deg = math.degrees(angle_rad)

        return {
            'distance': dist,
            'angle_deg': angle_deg,
        }

    def _update_pinch_gui_from_hand(self, hand_world_landmarks, hand_name: str):
        metrics = self._compute_pinch_metrics(hand_world_landmarks)
        if metrics is None:
            self._update_pinch_gui_no_hand()
            self._publish_pinch_state_none()
            return

        self.last_pinch_metrics = metrics
        self.last_pinch_hand = hand_name

        dist = metrics['distance']
        angle_deg = metrics['angle_deg']

        # Update live readouts
        self.pinch_distance_text.value = f'distance: {dist:.4f} m'
        self.pinch_angle_text.value = f'angle: {angle_deg:.1f} deg'

        # Defaults if not calibrated
        progress_val = -1.0
        progress_str = 'progress: -'
        state_str = 'state: UNCALIBRATED'
        closed_state = False

        if self.pinch_open_ref is not None and self.pinch_closed_ref is not None:
            open_d = self.pinch_open_ref['distance']
            closed_d = self.pinch_closed_ref['distance']

            # Guard against degenerate calibration
            denom = max(abs(open_d - closed_d), 1e-6)
            # We expect open_d > closed_d (open = fingers far apart, closed = pinched)
            progress = (open_d - dist) / denom
            progress = max(0.0, min(1.0, progress))
            progress_val = progress

            threshold = float(self.pinch_threshold_slider.value)
            closed_state = progress >= threshold

            progress_str = f'progress: {progress:.2f}'
            state = 'CLOSED' if closed_state else 'OPEN'
            state_str = f'state: {state} (thr={threshold:.2f})'

        # Update GUI
        self.pinch_progress_text.value = progress_str
        self.pinch_state_text.value = state_str

        # Publish pinch state to ROS
        self._publish_pinch_state(progress_val, closed_state, hand_name)

    def _update_pinch_gui_no_hand(self):
        self.pinch_distance_text.value = 'distance: -'
        self.pinch_angle_text.value = 'angle: -'
        self.pinch_progress_text.value = 'progress: -'
        self.pinch_state_text.value = 'state: NO HAND'

    def _publish_pinch_state(self, progress_val: float, closed_state: bool, hand_name: str):
        # progress_val: -1.0 if uncalibrated or no data, else [0,1]
        msg_p = Float32()
        msg_p.data = float(progress_val)
        self.pinch_progress_pub.publish(msg_p)

        msg_b = Bool()
        msg_b.data = bool(closed_state)
        self.pinch_closed_pub.publish(msg_b)

        msg_h = String()
        msg_h.data = hand_name
        self.pinch_hand_pub.publish(msg_h)

    def _publish_pinch_state_none(self):
        # No hand or no metrics: publish sentinel progress and "no hand"
        self._publish_pinch_state(progress_val=-1.0, closed_state=False, hand_name='')

    def _capture_pinch_reference(self, which: str):
        """
        Capture current pinch pose as 'open' or 'closed' reference.
        Uses the last computed pinch metrics (from whichever hand is active).
        """
        if self.last_pinch_metrics is None:
            self.get_logger().warn('Cannot capture pinch pose: no hand metrics available yet.')
            return

        if which == 'open':
            self.pinch_open_ref = dict(self.last_pinch_metrics)
            self.get_logger().info(
                f"Captured pinch OPEN pose: dist={self.pinch_open_ref['distance']:.4f} m, "
                f"angle={self.pinch_open_ref['angle_deg']:.1f} deg"
            )
        elif which == 'closed':
            self.pinch_closed_ref = dict(self.last_pinch_metrics)
            self.get_logger().info(
                f"Captured pinch CLOSED pose: dist={self.pinch_closed_ref['distance']:.4f} m, "
                f"angle={self.pinch_closed_ref['angle_deg']:.1f} deg"
            )
        else:
            self.get_logger().warn(f'Unknown pinch reference type: {which}')

    # ---------------- Cleanup ----------------
    def destroy_node(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MediapipeHandsViserRos()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
