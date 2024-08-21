import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from .inference_on_a_image import UniPoseLiveInferencer  # Adjust import as needed

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'jetson_webcam',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.br = CvBridge()
        
        # Initialize UniPoseInferencer with the model
        config_file = "/workspace/src/unipose/unipose/config_model/UniPose_SwinT.py"
        checkpoint_path = "/workspace/src/unipose/unipose/config_model/unipose_swint.pth"
        self.inferencer = UniPoseLiveInferencer(config_file, checkpoint_path, cpu_only=False)

    def listener_callback(self, data):
        self.get_logger().info('Receiving video frame')
        cv_image = self.br.imgmsg_to_cv2(data)
        
        # Run inference on the image
        instance_text_prompt = "person"  # Adjust as needed
        keypoint_text_example = "person"  # Adjust as needed
        
        output_image = self.inferencer.run_inference(
            cv_image=cv_image,
            instance_text_prompt=instance_text_prompt,
            keypoint_text_example=keypoint_text_example
        )

        # Display the output image
        cv2.imshow("camera", output_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
