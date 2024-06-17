import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
from tf2_ros import TransformBroadcaster 
from geometry_msgs.msg import TransformStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from ament_index_python.packages import get_package_share_directory
import os
import yaml
import numpy as np


class WebcamPublisher(Node):
    def __init__(self):
        super().__init__("webcam_publisher")

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.image_publisher = self.create_publisher(Image, "/camera", qos_profile=qos_profile)
        self.camera_info_publisher = self.create_publisher(CameraInfo, "/camera_info", 10)

        self.declare_parameter("camera_index", 0)
        self._cam_idx = (
            self.get_parameter("camera_index").get_parameter_value().integer_value
        )
        self.declare_parameter("camera_info_path", os.path.join(get_package_share_directory("moveicu_interfaces"), "models", "generic_camera_info.yaml"))
        cam_info_path = self.get_parameter("camera_info_path").get_parameter_value().string_value

        with open(cam_info_path, "r") as file_handle:
            calib_data = yaml.load(file_handle, Loader=yaml.FullLoader)

        self._camera_info_msg = CameraInfo()
        self._camera_info_msg.height = calib_data["image_height"]
        self._camera_info_msg.width = calib_data["image_width"]
        self._camera_info_msg.k = list(map(float, calib_data["camera_matrix"]["data"]))
        self._camera_info_msg.d = list(map(float, calib_data["distortion_coefficients"]["data"]))
        self._camera_info_msg.r = list(map(float, calib_data["rectification_matrix"]["data"]))
        self._camera_info_msg.p = list(map(float, calib_data["projection_matrix"]["data"]))
        self._camera_info_msg.distortion_model = calib_data["distortion_model"]
        self._camera_info_msg.roi.do_rectify = True

        self._cap = cv2.VideoCapture(self._cam_idx)
        self._cvbridge = CvBridge()

        self._tf_publisher = TransformBroadcaster(self)

        self.create_timer(1 / 60.0, self.publish_frame)  # 1/60.0 seconds



    def publish_frame(self):
        ret, frame = self._cap.read()
        if ret:
            img_msg = self._cvbridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header.frame_id = "camera"
            self.image_publisher.publish(img_msg)

            self._camera_info_msg.header.stamp = img_msg.header.stamp
            self._camera_info_msg.header.frame_id = img_msg.header.frame_id
            self.camera_info_publisher.publish(self._camera_info_msg)

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
