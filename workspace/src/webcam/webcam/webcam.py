import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from tf2_ros import TransformBroadcaster 
from geometry_msgs.msg import TransformStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


class WebcamPublisher(Node):
    def __init__(self):
        super().__init__("webcam_publisher")

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.publisher_ = self.create_publisher(Image, "/camera", qos_profile=qos_profile)
        self.declare_parameter("camera_index", 0)
        self._cam_idx = (
            self.get_parameter("camera_index").get_parameter_value().integer_value
        )
        self._cap = cv2.VideoCapture(self._cam_idx)
        self._cvbridge = CvBridge()

        self._tf_publisher = TransformBroadcaster(self)

        self.create_timer(1 / 60.0, self.publish_frame)  # 1/60.0 seconds



    def publish_frame(self):
        ret, frame = self._cap.read()
        if ret:
            img_msg = self._cvbridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header.frame_id = "camera"
            self.publisher_.publish(img_msg)

            trans = TransformStamped()
            trans.header.stamp = img_msg.header.stamp
            trans.header.frame_id = "world"
            trans.child_frame_id = "camera"
            trans.transform.translation.x = 0.0
            trans.transform.translation.y = 0.0
            trans.transform.translation.z = 0.0
            trans.transform.rotation.x = 0.0
            trans.transform.rotation.y = 0.0
            trans.transform.rotation.z = 0.0
            trans.transform.rotation.w = 1.0

            self._tf_publisher.sendTransform(trans)


def main(args=None):
    rclpy.init(args=args)
    this_node = WebcamPublisher()
    rclpy.spin(this_node)
    this_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
