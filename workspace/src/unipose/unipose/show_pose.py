import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from moveicu_interfaces.msg import StampedBoundingBoxList, StampedFacialLandmarksList
import message_filters

# Import your custom UniPose functions
from .inference_on_a_image import run_unipose_inference, load_model

class ShowPoseNode(Node):
    def __init__(self):
        super().__init__("show_pose")

        # QoS Profile for reliable message delivery
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Create subscriptions with QoS profile
        self.camera_sub = self.create_subscription(
            Image,
            '/camera',
            self.camera_callback,
            qos_profile=self.qos_profile
        )
        self.landmarks_sub = self.create_subscription(
            StampedFacialLandmarksList,
            '/landmarks',
            self.landmarks_callback,
            qos_profile=self.qos_profile
        )

        # Create message filters
        self.ts = message_filters.TimeSynchronizer([self.camera_sub, self.landmarks_sub], 10)
        self.ts.registerCallback(self.callback)

        # Publisher for the pose data
        self.pose_publisher = self.create_publisher(PoseArray, '/pose', 1)

        # CvBridge for image conversions
        self.cv_bridge = CvBridge()

        # Load UniPose Model
        config_file="/workspace/src/unipose/unipose/config_model/UniPose_SwinT.py",
        checkpoint_path="/workspace/src/unipose/unipose/config_model/unipose_swint.pth",
        self.unipose_model = load_model(config_file, checkpoint_path, cpu_only=False)

    def callback(self, camera_msg, landmarks_msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.cv_bridge.imgmsg_to_cv2(camera_msg, "bgr8")

        # Process the image with UniPose to get keypoints and bounding boxes
        _, keypoints_filt = run_unipose_inference(
            config_file="/workspace/src/unipose/unipose/config_model/UniPose_SwinT.py",
            checkpoint_path="/workspace/src/unipose/unipose/config_model/unipose_swint.pth",
            cv_image=cv_image,
            instance_text_prompt="person",
            keypoint_text_example="person"
        )

        # Create and populate PoseArray message
        pose_array_msg = PoseArray()
        poses = []
        for keypoint in keypoints_filt:
            pose = Pose()
            pose.position.x = keypoint[0]  # x coordinate
            pose.position.y = keypoint[1]  # y coordinate
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            poses.append(pose)
        pose_array_msg.poses = poses
        
        # Publish the PoseArray message
        self.pose_publisher.publish(pose_array_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ShowPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()