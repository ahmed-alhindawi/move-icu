import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class OverlayNode(Node):
    def __init__(self):
        super().__init__('overlay_node')
        
        # Subscribers to both the webcam and the pose estimate topics
        self.webcam_sub = self.create_subscription(
            Image, 'webcam', self.webcam_callback, 10)
        
        self.pose_sub = self.create_subscription(
            Image, 'pose_estimation', self.pose_callback, 10)
        
        # Publisher for the combined (overlayed) image
        self.publisher_ = self.create_publisher(Image, 'overlayed_image', 10)
        
        self.br = CvBridge()
        self.webcam_image = None
        self.pose_image = None
    
    def webcam_callback(self, data):
        self.webcam_image = self.br.imgmsg_to_cv2(data, "bgr8")
        self.try_overlay()
    
    def pose_callback(self, data):
        self.pose_image = self.br.imgmsg_to_cv2(data, "bgr8")
        self.try_overlay()
    
    def try_overlay(self):
        # Ensure both images are available
        if self.webcam_image is not None and self.pose_image is not None:
            
            # Resize pose image to match webcam image size if necessary
            if self.webcam_image.shape != self.pose_image.shape:
                self.pose_image = cv2.resize(self.pose_image, 
                                             (self.webcam_image.shape[1], self.webcam_image.shape[0]))

            # Create a mask for the non-white areas
            lower_white = np.array([200, 200, 200], dtype=np.uint8)  # Lower bound for white
            upper_white = np.array([255, 255, 255], dtype=np.uint8)  # Upper bound for white
            mask = cv2.inRange(self.pose_image, lower_white, upper_white)
            
            # Invert mask to get the non-white areas
            mask_inv = cv2.bitwise_not(mask)
            
            # Convert mask to 3-channel image
            mask_3ch = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
            
            # Prepare the pose image
            pose_image_bg = cv2.bitwise_and(self.pose_image, mask_3ch)
            
            # Prepare the background of the webcam image
            webcam_bg = cv2.bitwise_and(self.webcam_image, cv2.bitwise_not(mask_3ch))
            
            # Combine the two images
            overlayed_image = cv2.add(webcam_bg, pose_image_bg)
            
            # Publish the combined image
            overlayed_msg = self.br.cv2_to_imgmsg(overlayed_image, "bgr8")
            self.publisher_.publish(overlayed_msg)

def main(args=None):
    rclpy.init(args=args)
    overlay_node = OverlayNode()
    rclpy.spin(overlay_node)
    overlay_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

