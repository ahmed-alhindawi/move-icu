
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import numpy as np
  
class ImagePublisher(Node):
  
  def __init__(self):
    super().__init__('image_publisher')
    self.publisher_ = self.create_publisher(Image, 'jetson_webcam', 10)
    timer_period = 1/30.0
    self.timer = self.create_timer(timer_period, self.timer_callback)
    self.cap = cv2.VideoCapture(0)
    self.br = CvBridge()
    
  def timer_callback(self):
      ret, frame = self.cap.read()
      if not ret:
          self.get_logger().error('Failed to capture image')
          return
      try:
        frame = np.array(frame)
        # Convert the OpenCV image to a ROS Image message
        image_msg = self.br.cv2_to_imgmsg(frame, encoding="rgb8")
      
        # Publish the Image message
        self.publisher_.publish(image_msg)
        self.get_logger().info('Publishing video frame')

      except CvBridgeError as e:
         print(type(frame))
         print(e)
   
def main(args=None):
  rclpy.init(args=args)
  image_publisher = ImagePublisher()
  rclpy.spin(image_publisher)
  image_publisher.destroy_node()
  rclpy.shutdown()
   
if __name__ == '__main__':
  main()