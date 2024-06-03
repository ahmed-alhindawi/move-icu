import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class WebcamPublisher(Node):
    def __init__(self):
        super().__init__("webcam_publisher")
        self.publisher_ = self.create_publisher(Image, "/camera", 0)
        self.declare_parameter("camera_index", 0)
        self._cam_idx = (
            self.get_parameter("camera_index").get_parameter_value().integer_value
        )
        self._cap = cv2.VideoCapture(self._cam_idx)
        self._cvbridge = CvBridge()

        self.timer = self.create_timer(1 / 30.0, self.publish_frame)  # 1/30.0 seconds

    def publish_frame(self):
        ret, frame = self._cap.read()
        if ret:
            img_msg = self._cvbridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher_.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    this_node = WebcamPublisher()
    rclpy.spin(this_node)
    this_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
