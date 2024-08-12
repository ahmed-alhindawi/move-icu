from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # zed_launch,
        # Node(
        #     package='zed_wrapper',
        #     namespace='face1',
        #     executable='camera.launch.py'
        # ),
        Node(
            package='face_d',
            namespace='face1',
            executable='face_d',
            name='face_detector',
            remappings=[("/image_rect", "/zed/zed_node/rgb/image_rect_color"),
                        ("/faces", "/faces")]
        ),
        Node(
            package='unipose_d',
            namespace='face1',
            executable='unipose_d',
            name='unipose_detector',
            remappings=[("/camera", "/zed/zed_node/rgb/image_rect_color"),
                        ("/faces", "/faces")]
        ),
        Node(
            package='unipose_d',
            namespace='face1',
            executable='show_uniposes',
            name='show_uniposes',
            remappings=[("/camera", "/zed/zed_node/rgb/image_rect_color"),
                        ("/uniposes", "/uniposes"),
                        ("/faces", "/faces")]
        )
    ])