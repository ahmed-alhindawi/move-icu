from rclpy.node import Node
import cv2
from geometry_msgs.msg import PoseStamped
import numpy as np
import os
import ament_index_python
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec
from sensor_msgs.msg import CameraInfo
from image_geometry import PinholeCameraModel
from landmark_d.ros_np_multiarray import to_numpy_f64
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import transforms3d as t3d
import rclpy
from moveicu_interfaces.msg import StampedFacialLandmarksList, StampedBoundingBoxList
from tf2_ros import TransformBroadcaster 
from geometry_msgs.msg import TransformStamped
import message_filters


class FitLandmarks(Node):

    @staticmethod
    def wait_for_message(
        msg_type,
        node: 'Node',
        topic: str,
        time_to_wait=-1
    ):
        """
        Wait for the next incoming message.

        :param msg_type: message type
        :param node: node to initialize the subscription on
        :param topic: topic name to wait for message
        :param time_to_wait: seconds to wait before returning
        :returns: (True, msg) if a message was successfully received, (False, None) if message
            could not be obtained or shutdown was triggered asynchronously on the context.
        """
        context = node.context
        wait_set = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, context.handle)
        wait_set.clear_entities()

        sub = node.create_subscription(msg_type, topic, lambda _: None, 1)
        wait_set.add_subscription(sub.handle)
        sigint_gc = SignalHandlerGuardCondition(context=context)
        wait_set.add_guard_condition(sigint_gc.handle)

        timeout_nsec = timeout_sec_to_nsec(time_to_wait)
        wait_set.wait(timeout_nsec)

        subs_ready = wait_set.get_ready_entities('subscription')
        guards_ready = wait_set.get_ready_entities('guard_condition')

        if guards_ready:
            if sigint_gc.handle.pointer in guards_ready:
                raise ValueError("Shutdown triggered while waiting for message")

        if subs_ready:
            if sub.handle.pointer in subs_ready:
                msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
                if msg_info is not None:
                    return True, msg_info[0]

    def __init__(self):
        super().__init__('fit_landmarks')

        _qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        subscribers = [
            message_filters.Subscriber(self, StampedFacialLandmarksList, "/landmarks", qos_profile=qos_profile),
            message_filters.Subscriber(self, StampedBoundingBoxList, "/faces", qos_profile=qos_profile),
        ]
        self._ts = message_filters.TimeSynchronizer(subscribers, 5)
        self._ts.registerCallback(self.callback)
        self.tf_publisher = TransformBroadcaster(self)

        f68_fpath = os.path.join(ament_index_python.get_package_share_directory("moveicu_interfaces"), "models", "face_model_68.txt")

        with open(f68_fpath) as f:
            raw_values = f.readlines()
        
        self.model_points = np.array(raw_values, dtype=float).reshape((3, -1)).T

        self.img_proc = PinholeCameraModel()
        self.get_logger().info("Waiting for camera info")
        try:
            ret, cam_info = self.wait_for_message(CameraInfo, self, "/camera_info")
            self.img_proc.fromCameraInfo(cam_info)
            self.get_logger().info("...Done")
        except Exception as e:
            self.get_logger().error(f"Could not get camera info: {e}")
            self.destroy_node()
            return
        
    @staticmethod
    def _get_square_box(box):
        """Get a square box out of the given box, by expanding it."""
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

    def callback(self, ldmks_msg, bboxes_msg):
        camera_matrix = self.img_proc.intrinsicMatrix()
        dist_coeffs = self.img_proc.distortionCoeffs()
        img_width = self.img_proc.width
        img_height = self.img_proc.height

        for i, (ldmks_data, face_box) in enumerate(zip(ldmks_msg.data, bboxes_msg.data)):
            ldmks = to_numpy_f64(ldmks_data.landmarks)[..., :2]  # ignore the confidence values
            _diff_height_width = (face_box.ymax - face_box.ymin) - (
                face_box.xmax - face_box.xmin
            )
            _offset_y = int(abs(_diff_height_width / 2))
            _box_moved = self._move_box(face_box, [0, _offset_y])

            # Make box square.
            x1, y1, x2, y2 = self._get_square_box(_box_moved)
            # clamp to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_height, x2), min(img_width, y2)
            ldmks[..., 0] = (ldmks[..., 0] * (x2 - x1)) + (x1 + ((x2 - x1) / 2.0))
            ldmks[..., 1] = (ldmks[..., 1] * (y2 - y1)) + (y1 + ((y2 - y1) / 2.0))

            success, rodrigues_rotation, translation_vector, _ = cv2.solvePnPRansac(self.model_points,
                                                                                    ldmks.reshape(len(self.model_points), 1, 2),
                                                                                    cameraMatrix=camera_matrix,
                                                                                    distCoeffs=dist_coeffs, 
                                                                                    flags=cv2.SOLVEPNP_DLS)


            if not success:
                return False, None, None

            # this is generic point stabiliser, the underlying representation doesn't matter
            # rotation_vector, translation_vector = self.apply_kalman_filter_head_pose(0, rodrigues_rotation, translation_vector / 1000.0)

            # rotation_vector[0] += self.head_pitch

            rotation_matrix, _ = cv2.Rodrigues(rodrigues_rotation)
            rotation_matrix = np.matmul(rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
            m = np.zeros((4, 4))
            m[:3, :3] = rotation_matrix
            m[3, 3] = 1
            rpy_rotation = np.array(t3d.euler.mat2euler(m))

            translation_vector = (translation_vector / 1000.0).flatten()

            pose_msg = TransformStamped()
            pose_msg.header.stamp = ldmks_msg.header.stamp
            pose_msg.header.frame_id = "camera"
            pose_msg.child_frame_id = f"head_{i}"
            pose_msg.transform.translation.x = translation_vector[0]
            pose_msg.transform.translation.y = translation_vector[1]
            pose_msg.transform.translation.z = translation_vector[2]
            pose_msg.transform.rotation.x = rpy_rotation[0]
            pose_msg.transform.rotation.y = rpy_rotation[1]
            pose_msg.transform.rotation.z = rpy_rotation[2]
            pose_msg.transform.rotation.w = 1.0

            self.tf_publisher.sendTransform(pose_msg)


               
               
def main(args=None):
    rclpy.init(args=args)

    this_node = FitLandmarks()

    rclpy.spin(this_node)

    this_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               