from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Other nodes you may have
        Node(
            package='webcam',
            namespace='webcam_d',
            executable='webcam',
            name='webcam_publisher',
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
            package='landmark_d',
            namespace='face1',
            executable='landmark_d',
            name='landmark_detector',
            remappings=[("/camera", "/camera"),
                        ("/faces", "/faces")]
        ),
        Node(
            package='landmark_d',
            namespace='face1',
            executable='show_landmarks',
            name='show_landmarks',
            remappings=[("/camera", "/camera"),
                        ("/landmarks", "/landmarks"),
                        ("/faces", "/faces")]
        ),
        Node(
            package='unipose',
            namespace='unipose_namespace',  # Adjust this if needed
            executable='show_pose',
            name='show_pose',
            remappings=[("/camera", "/camera"),
                        ("/faces", "/faces"),
                        ("/landmarks", "/landmarks")]
        )
    ])
