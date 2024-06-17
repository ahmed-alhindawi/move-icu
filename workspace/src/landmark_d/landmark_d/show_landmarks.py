import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from moveicu_interfaces.msg import StampedBoundingBoxList, StampedFacialLandmarksList
import message_filters
from landmark_d.ros_np_multiarray import to_numpy_f64
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy 


class ShowLandmarks(Node):
    def __init__(self):
        super().__init__("show_landmarks")

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        subscribers = [
            message_filters.Subscriber(self, Image, "/camera", qos_profile=qos_profile),
            message_filters.Subscriber(self, StampedFacialLandmarksList, "/landmarks", qos_profile=qos_profile),
            message_filters.Subscriber(self, StampedBoundingBoxList, "/faces", qos_profile=qos_profile),
        ]
        self._ts = message_filters.TimeSynchronizer(subscribers, 5)
        self._ts.registerCallback(self.callback)

        self._publisher = self.create_publisher(Image, "/landmarks_image", 1)

        self._cvbridge = CvBridge()

    @staticmethod
    def _get_square_box(box):
        """Get a square box out of the >given box, by expanding it."""
        left_x, top_y, right_x, bottom_y = box

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:  # Already a square.
            return box
        elif diff > 0:  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:  # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def _move_box(box, offset):
        """Move the box to direction specified by vector offset"""
        left_x = box.xmin + offset[0]
        top_y = box.ymin + offset[1]
        right_x = box.xmax + offset[0]
        bottom_y = box.ymax + offset[1]

        return [left_x, top_y, right_x, bottom_y]

    def callback(self, img_msg, ldmks_msg, bboxes_msg):
        img = self._cvbridge.imgmsg_to_cv2(img_msg=img_msg, desired_encoding="bgr8")

        for ldmks_data, face_box in zip(ldmks_msg.data, bboxes_msg.data):
            ldmks = to_numpy_f64(ldmks_data.landmarks)
            _diff_height_width = (face_box.ymax - face_box.ymin) - (
                face_box.xmax - face_box.xmin
            )
            _offset_y = int(abs(_diff_height_width / 2))
            _box_moved = self._move_box(face_box, [0, _offset_y])

            # Make box square.
            x1, y1, x2, y2 = self._get_square_box(_box_moved)
            # clamp to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            ldmks[..., 0] = (ldmks[..., 0] * (x2 - x1)) + (x1 + ((x2 - x1) / 2.0))
            ldmks[..., 1] = (ldmks[..., 1] * (y2 - y1)) + (y1 + ((y2 - y1) / 2.0))
            img = self._draw_face(
                ldmks, img, face_box, draw_confidence=True, confidence_radius=3
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
