from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    image_rect_color_topic = "/zed/zed_node/rgb/image_rect_color"
    return LaunchDescription([
        Node(
            package='face_d',
            namespace='face1',
            executable='face_d',
            name='face_detector',
            remappings=[("/image_rect", image_rect_color_topic),
                        ("/faces", "/faces")]
        ),
        Node(
            package='landmark_d',
            namespace='face1',
            executable='landmark_d',
            name='landmark_detector',
            remappings=[("/camera", image_rect_color_topic),
                        ("/faces", "/faces")]
        ),
        Node(
            package='landmark_d',
            namespace='face1',
            executable='show_landmarks',
            name='show_landmarks',
            remappings=[("/camera", image_rect_color_topic),
                        ("/landmarks", "/landmarks"),
                        ("/faces", "/faces")]
        ),
        Node(
            package='unipose',
            namespace='face1',
            executable='show_pose',
            name='show_pose',
            remappings=[("/camera", image_rect_color_topic),
                        ("/unipose", "unipose")
                        ("/landmarks", "/landmarks"),
                        ("/faces", "/faces")]
        )
    ])