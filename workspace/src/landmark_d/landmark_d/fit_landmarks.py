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
from moveicu_interfaces.msg import StampedFacialLandmarksList
from tf2_ros import TransformBroadcaster 
from geometry_msgs.msg import TransformStamped

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
        super().__init__('threed_landmarks')

        _qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.subscriber_ = self.create_subscription(StampedFacialLandmarksList, "/landmarks", self.callback, qos_profile=_qos)
        self.publisher_ = self.create_publisher(PoseStamped, '/head_pose', qos_profile=_qos)
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

    def callback(self, ldmks_msg):
        camera_matrix = self.img_proc.intrinsicMatrix()
        dist_coeffs = self.img_proc.distortionCoeffs()

        for i, data in enumerate(ldmks_msg.data):
            landmarks = to_numpy_f64(data.landmarks)[..., :2]  # ignore the confidence values
            success, rodrigues_rotation, translation_vector, _ = cv2.solvePnPRansac(self.model_points,
                                                                                    landmarks.reshape(len(self.model_points), 1, 2),
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


               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               