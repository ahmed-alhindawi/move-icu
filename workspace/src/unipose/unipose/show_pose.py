#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from unipose.inference_on_a_image import run_unipose_inference  # Import the inference function

class ShowPose(Node):
    def __init__(self):
        super().__init__("show_pose")

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Subscriber for the image topic
        self._subscriber = message_filters.Subscriber(self, Image, "/camera", qos_profile=qos_profile)

        # Publisher for the annotated image
        self._publisher = self.create_publisher(Image, "/show_pose", 1)

        # CvBridge to convert between ROS and OpenCV
        self._cvbridge = CvBridge()

        # Time synchronizer to process images
        self._ts = message_filters.TimeSynchronizer([self._subscriber], 10)
        self._ts.registerCallback(self.callback)

    def callback(self, img_msg):
        try:
            # Convert the ROS Image message to a CV2 image
            cv_image = self._cvbridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridge.Error as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Run the pose estimation inference and get the annotated image
        annotated_image = run_unipose_inference(cv_image)

        try:
            # Convert the annotated CV image back to a ROS Image message
            output_image_msg = self._cvbridge.cv2_to_imgmsg(annotated_image, "bgr8")
            self._publisher.publish(output_image_msg)
        except CvBridge.Error as e:
            self.get_logger().error(f"CvBridge Error: {e}")

def main(args=None):
    rclpy.init(args=args)

    node = ShowPose()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
