from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mediapipe_hands_viser',
            executable='mediapipe_hands_viser_node',
            name='mediapipe_hands_viser',
            output='screen',
            parameters=[
                # Example parameters you could extend later:
                # {'camera_index': 0},
                # {'publish_pose_array': True},
            ]
        )
    ])
