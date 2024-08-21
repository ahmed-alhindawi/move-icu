#!/usr/bin/env python3

import rclpy
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from unipose.inference_on_a_image import run_unipose_inference  # Import your pose estimation function

class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')
        
        # Initialize publisher
        self.publisher = self.create_publisher(Image, '/pose_estimation/output_image', 10)
        
        # Initialize subscriber
        self.subscription = self.create_subscription(
            Image,
            '/camera',  # Change this to your actual camera topic if different
            self.image_callback,
            10
        )
        
        # Initialize CvBridge
        self.br = CvBridge()
        
        # Timer for publishing images (if needed for other use cases)
        # self.timer = self.create_timer(1.0, self.publish_image)  # Optional

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.br.imgmsg_to_cv2(msg)
        
        # Apply pose estimation
        try:
            output_image = run_unipose_inference(cv_image)
            self.get_logger().info('Pose estimation applied successfully.')
        except Exception as e:
            self.get_logger().error(f'Error during pose estimation: {str(e)}')
            output_image = cv_image  # Fallback to original image in case of error
        
        # Convert OpenCV image back to ROS Image message
        try:
            ros_image = self.br.cv2_to_imgmsg(output_image, encoding='bgr8')
            self.publisher.publish(ros_image)
            self.get_logger().info('Published processed image to /pose_estimation/output_image')
        except Exception as e:
            self.get_logger().error(f'Error converting image to ROS message: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()
    rclpy.spin(node)
    
    # Clean up
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
