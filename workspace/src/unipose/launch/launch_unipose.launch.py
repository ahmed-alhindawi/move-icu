from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch img_pub node
        Node(
            package='unipose',
            executable='img_pub',
            name='img_pub_node',
            remappings=[('/webcam', '/webcam')]  # Adjust remappings if needed
        ),

        # Launch img_sub node
        Node(
            package='unipose',
            executable='img_sub',
            name='img_sub_node',
            remappings=[('/pose_estimation', '/pose_estimation')]  # Adjust remappings if needed
        ),

        # Launch img_pose_webcam node
        Node(
            package='unipose',
            executable='img_pose_webcam',
            name='img_pose_webcam_node',
            remappings=[('/overlayed_image', '/overlayed_image')]  # Adjust remappings if needed
        )
    ])
