from rclpy.node import Node
from sensor_msgs.msg import Image
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
from ros_np_multiarray import to_numpy_f64

class ShowLandmarks(Node):

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
        try:
            wait_set.add_subscription(sub.handle)
            sigint_gc = SignalHandlerGuardCondition(context=context)
            wait_set.add_guard_condition(sigint_gc.handle)

            timeout_nsec = timeout_sec_to_nsec(time_to_wait)
            wait_set.wait(timeout_nsec)

            subs_ready = wait_set.get_ready_entities('subscription')
            guards_ready = wait_set.get_ready_entities('guard_condition')

            if guards_ready:
                if sigint_gc.handle.pointer in guards_ready:
                    return False, None

            if subs_ready:
                if sub.handle.pointer in subs_ready:
                    msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
                    if msg_info is not None:
                        return True, msg_info[0]
        finally:
            node.destroy_subscription(sub)

        return False, None

    def __init__(self):
        super().__init__('face_detector')
        self.subscriber_ = self.create_subscription(Image, "/landmarks", self.callback, 10)
        self.publisher_ = self.create_publisher(PoseStamped, '/head_pose', 10)

        f68_fpath = os.path.join(ament_index_python.get_package_share_directory("cartesian_interfaces"), "models", "face_68.txt")

        with open(f68_fpath) as f:
            raw_values = f.readlines()
        
        self.model_points = np.array(raw_values).reshape((3, -1)).T

        self.img_proc = PinholeCameraModel()
        ret, cam_info = self.wait_for_message(CameraInfo, self, "/camera_info")
        if ret:
            self.img_proc.fromCameraInfo(cam_info)
        else:
            raise ValueError("Unable to get CameraInfo message")

    def callback(self, ldmks_msg):
        camera_matrix = self.img_proc.intrinsicMatrix()
        dist_coeffs = self.img_proc.distortionCoeffs()

        for data in ldmks_msg.data:
            landmarks = to_numpy_f64(data.landmarks)
            success, rodrigues_rotation, translation_vector, _ = cv2.solvePnPRansac(self.model_points,
                                                                                    landmarks.reshape(len(self.model_points), 1, 2),
                                                                                    cameraMatrix=camera_matrix,
                                                                                    distCoeffs=dist_coeffs, 
                                                                                    flags=cv2.SOLVEPNP_DLS)


            if not success:
                return False, None, None

            # this is generic point stabiliser, the underlying representation doesn't matter
            rotation_vector, translation_vector = self.apply_kalman_filter_head_pose(0, rodrigues_rotation, translation_vector / 1000.0)

            rotation_vector[0] += self.head_pitch

            _rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            _rotation_matrix = np.matmul(_rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
            _m = np.zeros((4, 4))
            _m[:3, :3] = _rotation_matrix
            _m[3, 3] = 1
            _rpy_rotation = np.array(_euler_from_matrix(_m))

            return success, _rpy_rotation, translation_vector
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               