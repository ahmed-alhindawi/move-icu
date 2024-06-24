from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # zed_launch,
        Node(
            package='zed_wrapper',
            namespace='face1',
            executable='camera.launch.py'
        ),
        Node(
            package='face_d',
            namespace='face1',
            executable='face_d',
            name='face_detector',
            remappings=[("/camera", "/zed/zed_node/rgb/image_rect_color"),
                        ("/faces", "/faces")]
        ),
        Node(
            package='landmark_d',
            namespace='face1',
            executable='landmark_d',
            name='landmark_detector',
            remappings=[("/camera", "/zed/zed_node/rgb/image_rect_color"),
                        ("/faces", "/faces")]
        ),
        Node(
            package='landmark_d',
            namespace='face1',
            executable='show_landmarks',
            name='show_landmarks',
            remappings=[("/camera", "/zed/zed_node/rgb/image_rect_color"),
                        ("/landmarks", "/landmarks"),
                        ("/faces", "/faces")]
        )
    ])