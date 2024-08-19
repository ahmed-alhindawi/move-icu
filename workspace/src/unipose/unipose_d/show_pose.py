# Import necessary ROS and OpenCV libraries
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from PIL import Image as PILImage
import numpy as np

# Import your inference function
from inference_on_a_image import run_unipose_inference



class ShowPose:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('unipose_node', anonymous=True)

        # Create a CvBridge object to convert ROS images to OpenCV images
        self.bridge = CvBridge()

        # Subscribe to the camera/image topic (replace with your actual topic)
        self.image_sub = rospy.Subscriber("/cameracd ..", Image, self.callback)

        # Optional: If you want to publish processed images
        self.image_pub = rospy.Publisher("/unipose/output_image", Image, queue_size=10)

    def callback(self, ros_image):
        try:
            # Convert the ROS Image message to a PIL image
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # Run the inference on the image
            processed_image_pil = run_unipose_inference(
                config_file="path_to_config_file",
                checkpoint_path="path_to_checkpoint_file",
                image_pil=pil_image,
                instance_text_prompt="person",  # Example prompt
                output_dir="/tmp",
                keypoint_text_example=None,
                box_threshold=0.1,
                iou_threshold=0.9,
                cpu_only=False
            )

            # Convert processed PIL image back to OpenCV format
            processed_cv_image = cv2.cvtColor(np.array(processed_image_pil), cv2.COLOR_RGB2BGR)
            
            # Optional: Publish the processed image
            try:
                processed_ros_image = self.bridge.cv2_to_imgmsg(processed_cv_image, encoding="bgr8")
                self.image_pub.publish(processed_ros_image)
            except CvBridgeError as e:
                print(e)

        except CvBridgeError as e:
            print(e)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = UniPoseNode()
    node.run()
