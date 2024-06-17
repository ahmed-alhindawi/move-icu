import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from moveicu_interfaces.msg import BoundingBox, StampedBoundingBoxList
from face_d.SFD import sfd_detector
from ament_index_python.packages import get_package_share_directory
import os
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


class FaceDetector(Node):

    def __init__(self):
        super().__init__('face_detector')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.subscriber_ = self.create_subscription(Image, "/camera", self.callback, qos_profile=qos_profile)
        self.publisher_ = self.create_publisher(StampedBoundingBoxList, '/faces', qos_profile=qos_profile)
        path_to_model = os.path.join(get_package_share_directory("moveicu_interfaces"), "models", "s3fd_facedetector.ckpt")
        self.get_logger().info(f"Loading model from {path_to_model}")

        self._sfd = sfd_detector.SFDDetector(device="cuda:0", path_to_detector=path_to_model)
        self._cvbridge = CvBridge()

    def callback(self, img_msg):
        img = self._cvbridge.imgmsg_to_cv2(img_msg=img_msg, desired_encoding="bgr8")
        img_small = cv2.resize(img, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
        bbs = self._sfd.detect_from_image(img_small)
        face_bbs, confidences = bbs[:, :4], bbs[:, 4]
        face_bbs *= 4.0

        boxes = []
        for face_bb, conf in zip(face_bbs, confidences):
            face_box = face_bb.cpu().tolist()
            box = BoundingBox()
            box.xmin = face_box[0]
            box.ymin = face_box[1]
            box.xmax = face_box[2]
            box.ymax = face_box[3]
            box.confidence = float(conf)
            boxes.append(box)

        msg = StampedBoundingBoxList()
        msg.header = img_msg.header
        msg.data = boxes
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    this_node = FaceDetector()

    rclpy.spin(this_node)

    this_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()