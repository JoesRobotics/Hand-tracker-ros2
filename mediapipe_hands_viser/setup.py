from setuptools import setup

package_name = 'mediapipe_hands_viser'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/mediapipe_hands_viser.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='MediaPipe Hands to ROS 2 bridge with Viser 3D visualization.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mediapipe_hands_viser_node = mediapipe_hands_viser.mediapipe_hands_viser_node:main',
        ],
    },
)
