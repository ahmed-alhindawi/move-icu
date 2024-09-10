import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
from .inference_on_a_image import UniPoseLiveInferencer  # Adjust import as needed

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'webcam',
            self.listener_callback,
            1)  # Increased queue size
        self.publisher = self.create_publisher(Image, 'pose_estimation', 1)  # Increased queue size
        self.br = CvBridge()
        
        # Initialize UniPoseInferencer with the model
        config_file = "/workspace/src/unipose/unipose/config_model/UniPose_SwinT.py"
        checkpoint_path = "/workspace/src/unipose/unipose/config_model/unipose_swint.pth"
        self.inferencer = UniPoseLiveInferencer(config_file, checkpoint_path, cpu_only=False)

    def listener_callback(self, data):
        # Calculate the delay

        #import pdb;pdb.set_trace()
        cv_image = self.br.imgmsg_to_cv2(data)

        # Resize the image if needed (optional)
        resized_image = cv2.resize(cv_image, (640, 480))

        start_time = time.time()
        output_image = self.inferencer.run_inference(cv_image=resized_image)
        end_time = time.time()

        delay_2 = end_time - start_time

        self.get_logger().info(f'Processing time: {delay_2:.2f} seconds')
        

        output_image_msg = self.br.cv2_to_imgmsg(output_image, "bgr8")
        self.publisher.publish(output_image_msg)



def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
