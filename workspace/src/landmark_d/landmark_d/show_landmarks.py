import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from moveicu_interfaces.msg import StampedBoundingBoxList, StampedFacialLandmarksList
from message_filters import TimeSynchronizer, Subscriber
from landmark_d.ros_np_multiarray import to_numpy_f64
import numpy as np


class ShowLandmarks(Node):
    def __init__(self):
        super().__init__("face_detector")

        subscribers = [
            Subscriber(self, Image, "/camera"),
            Subscriber(self, StampedFacialLandmarksList, "/landmarks"),
            Subscriber(self, StampedBoundingBoxList, "/faces"),
        ]
        self._ts = TimeSynchronizer(subscribers, 5)
        self._ts.registerCallback(self.callback)

        self._publisher = self.create_publisher(Image, "/landmarks_image", 10)

        self._cvbridge = CvBridge()

    def callback(self, img_msg, ldmks_msg, bboxes_msg):
        img = self._cvbridge.imgmsg_to_cv2(img_msg=img_msg, desired_encoding="bgr8")

        for ldmks_data, bbox_data in zip(ldmks_msg.data, bboxes_msg.data):
            ldmks = to_numpy_f64(ldmks_data.landmarks)
            img = self._draw_face(
                ldmks, img, bbox_data, draw_confidence=True, confidence_radius=2
            )

        img_msg = self._cvbridge.cv2_to_imgmsg(img, encoding="bgr8")
        self._publisher.publish(img_msg)

    @staticmethod
    def _draw_face(
        ldmks,
        img,
        face_box,
        color=(255, 255, 255),
        draw_confidence=False,
        confidence_radius=2,
    ):
        ldmks_np = ldmks.astype(int)
        chin = ldmks_np[0:17, :2]
        left_brow = ldmks_np[22:27, :2]
        right_brow = ldmks_np[17:22, :2]
        right_eye = ldmks_np[36:42, :2]
        left_eye = ldmks_np[42:48, :2]
        nose = ldmks_np[27:31, :2]
        lower_nose = ldmks_np[30:36, :2]
        mouth = ldmks_np[48:60, :2]
        mouth2 = ldmks_np[60:69, :2]
        img = cv2.polylines(
            img, [chin], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA
        )
        img = cv2.polylines(
            img,
            [left_brow],
            isClosed=False,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        img = cv2.polylines(
            img,
            [right_brow],
            isClosed=False,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        img = cv2.polylines(
            img,
            [right_eye],
            isClosed=True,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        img = cv2.polylines(
            img,
            [left_eye],
            isClosed=True,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        img = cv2.polylines(
            img, [nose], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA
        )
        img = cv2.polylines(
            img,
            [lower_nose],
            isClosed=True,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        img = cv2.polylines(
            img, [mouth], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA
        )
        img = cv2.polylines(
            img, [mouth2], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA
        )

        if draw_confidence:
            variance = ldmks[:, 2]
            for mark, log_var in zip(ldmks_np, variance):
                radius = int(
                    np.sqrt(log_var)
                    * (face_box.xmax - face_box.xmin)
                    * confidence_radius
                )
                img = cv2.circle(
                    img,
                    center=(int(mark[0]), int(mark[1])),
                    color=color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                    radius=radius,
                )

        return img


def main(args=None):
    rclpy.init(args=args)

    this_node = ShowLandmarks()

    rclpy.spin(this_node)

    this_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
