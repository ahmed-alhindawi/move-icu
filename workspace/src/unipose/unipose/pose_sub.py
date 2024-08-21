import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from .inference_on_a_image import run_unipose_inference
  
class ImageSubscriber(Node):
  
  def __init__(self):
    super().__init__('image_subscriber')
    self.subscription = self.create_subscription(
      Image, 
      'jetson_webcam', 
      self.listener_callback, 
      10)
    self.subscription # prevent unused variable warning
    self.br = CvBridge()
    
  def listener_callback(self, data):
    self.get_logger().info('Receiving video frame')
    current_frame = self.br.imgmsg_to_cv2(data)

    pose_frame = run_unipose_inference(
            config_file="/workspace/src/unipose/unipose/config_model/UniPose_SwinT.py",
            checkpoint_path="/workspace/src/unipose/unipose/config_model/unipose_swint.pth",
            cv_image=current_frame,
            instance_text_prompt="person",
            keypoint_text_example="person"
        )


    # # Apply the pose estimation function
    # pose_frame = run_unipose_inference(current_frame)

    cv2.imshow("camera", pose_frame)
    cv2.waitKey(1)
   
def main(args=None):
  rclpy.init(args=args)
  image_subscriber = ImageSubscriber()
  rclpy.spin(image_subscriber)
  image_subscriber.destroy_node()
  rclpy.shutdown()
   
if __name__ == '__main__':
  main()