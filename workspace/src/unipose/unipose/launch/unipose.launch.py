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
            remappings=[("/camera", "/camera")]
        ),
        Node(
            package='unipose',
            namespace='unipose',
            executable='show_pose',
            name='show_pose',
            remappings=[
                ("/camera", "/camera"),
                ("/unipose", "/unipose")
            ]
        )
    ])
