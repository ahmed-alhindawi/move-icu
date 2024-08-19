#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from unipose.inference_on_a_image import run_unipose_inference  # Import the inference function

class ShowPose(Node):
    def __init__(self):
        super().__init__('unipose')
        
        # Subscribe to the camera topic
        self.image_sub = self.create_subscription(
            Image,
            '/camera',  # Adjust the topic name as needed
            self.image_callback,
            10)
        
        # Publisher for the annotated image
        self.image_pub = self.create_publisher(Image, '/unipose/output_image', 10)
        
        # Set up the OpenCV bridge
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to a CV2 image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridge.Error as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Run the inference using the imported function
        label = run_unipose_inference(cv_image)

        # Display the label on the image
        cv2.putText(cv_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        try:
            # Convert the annotated CV image back to a ROS Image message
            output_image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            
            # Publish the output image
            self.image_pub.publish(output_image_msg)
        except CvBridge.Error as e:
            self.get_logger().error(f"CvBridge Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    node = ShowPose()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
