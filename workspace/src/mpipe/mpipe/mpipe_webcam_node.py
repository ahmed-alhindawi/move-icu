import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time

class WebcamImagePublisher(Node):
    def __init__(self):
        super().__init__('webcam_image_publisher')
        self.publisher_ = self.create_publisher(Image, 'videostream', 10)
        #self.timer = self.create_timer(1, self.timer_callback)
        self.cap = cv2.VideoCapture(0)
        self.bridge = CvBridge()

        # self.last_time = time.time()
        # self.frame_count = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                    print("Error: Failed to cappture frame.")
                    break
            
            try:
                img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.publisher_.publish(img_msg)
                #self.get_logger().info('Publishing video frame')

                # self.frame_count += 1
                # current_time = time.time()
                # elapsed_time = current_time - self.last_time

                # if elapsed_time >= 1.0:
                #      fps = self.frame_count / elapsed_time
                #      self.get_logger().info(f"FPS: {fps:.2f}")
                #      self.frame_count = 0
                #      self.last_time = current_time
            except CvBridgeError as error:
                 print(error)


#    def timer_callback(self):
#        ret, frame = self.cap.read()
#        if not ret:
#            print("Error: Failed to capture frame.")
#            return
#        
#        #import pdb;pdb.set_trace()
#        frame = np.array(frame)
#        # Convert the OpenCV image to a ROS Image message
#        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
#        
#        # Publish the Image message
#        self.publisher_.publish(image_msg)
#        #self.get_logger().info('Publishing video frame')
    
def main(args=None):
    rclpy.init(args=args)
    node = WebcamImagePublisher()
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
