import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time

class WebcamImagePublisher(Node):
    # Node that publishes raw webcam images to the 'videostream' topic

    def __init__(self):
        super().__init__('webcam_image_publisher')
        self.publisher_ = self.create_publisher(Image, 'videostream', 10)
        self.cap = cv2.VideoCapture(0)
        self.bridge = CvBridge()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                    print("Error: Failed to cappture frame.")
                    break
            
            try:
                img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.publisher_.publish(img_msg)

            except CvBridgeError as error:
                 print(error)
            
def main(args=None):
    rclpy.init(args=args)
    node = WebcamImagePublisher()
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
