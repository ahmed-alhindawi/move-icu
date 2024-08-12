from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='webcam',
            namespace='webcam_d',
            executable='webcam',
            name='wecam_publisher',
            remappings=[("/camera", "/camera"),
                        ("/faces", "/faces")]
        ),
        Node(
            package='face_d',
            namespace='face1',
            executable='face_d',
            name='face_detector',
            remappings=[("/camera", "/camera"),
                        ("/faces", "/faces")]
        ),
        Node(
            package='unipose_d',
            namespace='face1',
            executable='unipose_d',
            name='pose_detector',
            remappings=[("/camera", "/camera"),
                        ("/faces", "/faces")]
        ),
        Node(
            package='unipose_d',
            namespace='face1',
            executable='show_pose',
            name='show_pose',
            remappings=[("/camera", "/camera"),
                        ("/unipose", "/unipose"),
                        ("/faces", "/faces")]
        )
    ])