import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import time
import cv2

import mediapipe as mp

class MediaPipePoseEstimator(Node):
    '''
        Node that processes webcam frames to detect key body locations.
        Subscribes to 'videostream' topic (raw webcam images)
        Publishes to 'pose_skeleton' topic (webcam images overlaid with detected landmarks/skeleton)
    '''

    def __init__(self):
        super().__init__('mpipe_pose_estimator')

        # Import mediapipe model and drawing utilities
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # Load mediapipe model
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.subscription = self.create_subscription(
            Image,
            'videostream',
            self.listener_callback,
            1)
        self.publisher= self.create_publisher(Image, 'pose_skeleton', 10)
        self.bridge = CvBridge()


    def listener_callback(self, data):
        
        # start_time = time.time()

        # Convert image to CV2 
        cv_frame = self.bridge.imgmsg_to_cv2(data)

        # Recolor image to RGB
        image = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = self.pose.process(image)

        # Color back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        #try:
        #    landmarks = results.pose_landmarks.landmark
        #except:
        #    pass

        # Render detections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Calculate the delay
        #end_time = time.time()
        #delay = end_time - start_time
        #self.get_logger().info(f'Processing time: {delay:.2f} seconds')
        
        output_image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.publisher.publish(output_image_msg)
        #self.get_logger().info('Publishing pose estimation image')


def main(args=None):
    rclpy.init(args=args)
    node = MediaPipePoseEstimator()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()