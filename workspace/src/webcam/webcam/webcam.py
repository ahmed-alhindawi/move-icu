import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class WebcamPublisher(Node):
    
        def __init__(self):
            super().__init__('webcam_publisher')
            self.publisher_ = self.create_publisher(Image, '/camera', 10)
            self.declare_parameter('camera_index', 0)
            self._cvbridge = CvBridge()
    
        def publish_frame(self):
            cam_idx = self.get_parameter('camera_index').get_parameter_value().integer_value
            cap = cv2.VideoCapture(cam_idx)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    img_msg = self._cvbridge.cv2_to_imgmsg(frame, encoding="bgr8")
                    self.publisher_.publish(img_msg)
                else:
                    print("Error reading frame")


def main(args=None):
    rclpy.init(args=args)
    this_node = WebcamPublisher()
    this_node.publish_frame() 
    this_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()