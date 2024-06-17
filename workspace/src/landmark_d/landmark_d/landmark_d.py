from glob import glob
import ament_index_python
import rclpy
import rclpy.logging
from rclpy.node import Node
import torch
import albumentations as albu
import albumentations.pytorch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from moveicu_interfaces.msg import (
    StampedBoundingBoxList,
    StampedFacialLandmarksList,
    FacialLandmarks,
)
import os
from message_filters import TimeSynchronizer, Subscriber
from landmark_d.LandmarkEstimationResNet import LandmarkEstimationResNet
from landmark_d.ros_np_multiarray import to_multiarray_f64
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


class LandmarkExtractor(Node):
    def __init__(self):
        super().__init__("face_detector")

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        subscribers = [
            Subscriber(self, Image, "/camera", qos_profile=qos_profile),
            Subscriber(self, StampedBoundingBoxList, "/faces", qos_profile=qos_profile),
        ]
        self._ts = TimeSynchronizer(subscribers, 5)
        self._ts.registerCallback(self.callback)

        modelpaths = glob(
            os.path.join(
                ament_index_python.get_package_share_directory("moveicu_interfaces"),
                "models",
                "landmark_*.ckpt",
            )
        )
        self.get_logger().info(
            f"Found {len(modelpaths)} models for landmark extraction, Loading..."
        )

        self.models = []
        for path in modelpaths:
            self.get_logger().info(f"Loading: {path}")
            self.models.append(self._load_network(path))
            # self.models = [self._load_network(path) for path in modelpaths]
        self.get_logger().info("...Done")

        self.landmark_extractor_transform = albu.Compose(
            [
                albu.Resize(height=112, width=112),
                albu.Normalize(),
                albu.pytorch.transforms.ToTensorV2(),
            ]
        )

        self._cvbridge = CvBridge()

        self.publisher_ = self.create_publisher(
            StampedFacialLandmarksList, "/landmarks", 10
        )

    def callback(self, img_msg, bboxes_msg):
        img = self._cvbridge.imgmsg_to_cv2(img_msg=img_msg, desired_encoding="bgr8")

        # pick the face with the biggest confidence
        ldmks = self._get_landmarks(
            img,
            bboxes_msg.data,
            landmark_extractors=self.models,
            transform=self.landmark_extractor_transform,
            device="cuda:0",
        )
        if ldmks is None:
            return

        ldmks = ldmks.numpy()
        ldmks_msgs = []
        for i in range(len(bboxes_msg.data)):
            ldmks_msg = FacialLandmarks()
            ldmks_msg.landmarks = to_multiarray_f64(ldmks[i, ...])
            ldmks_msgs.append(ldmks_msg)

        msg = StampedFacialLandmarksList()
        msg.header = img_msg.header
        msg.data = ldmks_msgs

        self.publisher_.publish(msg)

    @staticmethod
    def _load_jit_network(path):
        return torch.jit.load(path)

    @staticmethod
    def _load_network(path, num_out=3):
        model_params = torch.load(path)["state_dict"]

        model_prefix = "model."
        state_dict = {
            k[len(model_prefix) :]: v
            for k, v in model_params.items()
            if k.startswith(model_prefix)
        }

        model = LandmarkEstimationResNet(
            backbone=LandmarkEstimationResNet.ResNetBackbone.Resnet18, num_out=num_out
        )
        model.load_state_dict(state_dict)
        model.eval()
        model.to("cuda:0")

        return model

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

    def _get_landmarks(
        self, frame, face_boxes, landmark_extractors, transform, device="cuda:0"
    ):
        if len(face_boxes) < 1:
            return None

        face_imgs = []
        for face_box in face_boxes:
            _diff_height_width = (face_box.ymax - face_box.ymin) - (
                face_box.xmax - face_box.xmin
            )
            _offset_y = int(abs(_diff_height_width / 2))
            _box_moved = self._move_box(face_box, [0, _offset_y])

            # Make box square.
            x1, y1, x2, y2 = self._get_square_box(_box_moved)
            # clamp to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            img = frame[int(y1) : int(y2), int(x1) : int(x2), :]  # shape: HWC

            transformed_img = transform(image=img)
            transformed_img = transformed_img["image"].unsqueeze(0)
            face_imgs.append(transformed_img)

        transformed_imgs = torch.vstack(face_imgs).to(device)

        num_models = len(landmark_extractors)

        with torch.no_grad():
            outputs = (
                torch.vstack(
                    [model(transformed_imgs) for model in landmark_extractors]
                )
                .reshape(num_models, -1, 68, 3)
                .cpu()
            )
            outputs[:, :, :, 2] = torch.exp(
                outputs[:, :, :, 2]
            )  # models x batch x landmarks x dims

        sum_variances = 1.0 / (1.0 / outputs[..., 2]).sum(dim=0).reshape(
            -1, 68, 1
        )  # imgs, landmarks, variance
        output = (
            (outputs[..., :2] / (outputs[..., 2]).reshape(num_models, -1, 68, 1)).sum(
                dim=0
            )
        ) * sum_variances
        result = torch.cat([output, sum_variances], dim=-1)

        return result


def main(args=None):
    rclpy.init(args=args)

    this_node = LandmarkExtractor()

    rclpy.spin(this_node)

    this_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
