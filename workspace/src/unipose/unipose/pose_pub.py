
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
  
class ImagePublisher(Node):
  
  def __init__(self):
    super().__init__('image_publisher')
    self.publisher_ = self.create_publisher(Image, 'webcam', 1)
    timer_period = 0.1
    self.timer = self.create_timer(timer_period, self.timer_callback)
    self.cap = cv2.VideoCapture(0)
    self.br = CvBridge()
    
  def timer_callback(self):
      ret, frame = self.cap.read()
      if not ret:
          self.get_logger().error('Failed to capture image')
          return
      
      # Convert the OpenCV image to a ROS Image message
      image_msg = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
      
      # Publish the Image message
      self.publisher_.publish(image_msg)
      self.get_logger().info('Publishing video frame')
   
def main(args=None):
  rclpy.init(args=args)
  image_publisher = ImagePublisher()
  rclpy.spin(image_publisher)
  image_publisher.destroy_node()
  rclpy.shutdown()
   
if __name__ == '__main__':
  main()