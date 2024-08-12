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
            package='unipose_d',
            namespace='face1',
            executable='unipose_d',
            name='pose_detector',
            remappings=[("/camera", image_rect_color_topic),
                        ("/faces", "/faces")]
        ),
        Node(
            package='unipose_d',
            namespace='face1',
            executable='show_pose',
            name='show_pose',
            remappings=[("/camera", image_rect_color_topic),
                        ("/unipose", "/unipose"),
                        ("/faces", "/faces")]
        )
    ])