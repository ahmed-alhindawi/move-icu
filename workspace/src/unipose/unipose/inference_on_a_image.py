import sys
sys.path.insert(1, '.')

import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import clip
import torchvision.transforms as T
from src.unipose.unipose.models import build_model
from src.unipose.unipose.predefined_keypoints import *
from src.unipose.unipose.util import box_ops
from src.unipose.unipose.util.config import Config
from src.unipose.unipose.util.utils import clean_state_dict
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib import transforms
from torchvision.ops import nms
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

def text_encoding(instance_names, keypoints_names, model, device):
    def encode_text(descriptions):
        texts = clip.tokenize(descriptions).to(device)
        with torch.no_grad():
            features = model.encode_text(texts)  # Shape: (N, 512)
        return features

    # Only focusing on "person" for instance and keypoints
    instance_descriptions = [f"a photo of {cat.lower().replace('_', ' ').replace('-', ' ')}" for cat in instance_names]
    kpt_descriptions = [f"a photo of {kpt.lower().replace('_', ' ')}" for kpt in keypoints_names]

    # Encode all descriptions at once
    ins_text_embeddings = encode_text(instance_descriptions)
    kpt_text_embeddings = encode_text(kpt_descriptions)

    return ins_text_embeddings, kpt_text_embeddings



def plot_on_image(image_pil, tgt, keypoint_skeleton, keypoint_text_prompt):
    num_kpts = len(keypoint_text_prompt)
    H, W = tgt["size"]

    # Convert PIL image to a NumPy array (OpenCV format)
    image_np = np.array(image_pil)

    color_kpt = [
        (0, 0, 0), (255, 255, 255), (255, 0, 0),
        (255, 255, 0), (128, 41, 41), (0, 0, 255),
        (176, 225, 230), (0, 255, 0), (161, 33, 240),
        (209, 181, 140), (255, 97, 0), (135, 38, 87),
        (255, 100, 71), (255, 0, 255), (10, 23, 69),
        (51, 161, 201), (240, 230, 140), (85, 107, 46),
        (135, 207, 235), (181, 125, 220), (64, 224, 209)
    ]
    color_box = (135, 207, 235)

    # Draw bounding boxes
    for box in tgt['boxes'].cpu():
        unnormbbox = box * torch.Tensor([W, H, W, H])
        unnormbbox[:2] -= unnormbbox[2:] / 2
        bbox_x, bbox_y, bbox_w, bbox_h = unnormbbox.tolist()
        top_left = (int(bbox_x), int(bbox_y))
        bottom_right = (int(bbox_x + bbox_w), int(bbox_y + bbox_h))
        cv2.rectangle(image_np, top_left, bottom_right, color_box, 1)
    
    if 'keypoints' in tgt:
        sks = np.array(keypoint_skeleton)
        if sks.min() == 1:
            sks -= 1

        for idx, ann in enumerate(tgt['keypoints']):
            kp = np.array(ann.cpu())
            Z = kp[:num_kpts * 2] * np.array([W, H] * num_kpts)
            x = Z[0::2]
            y = Z[1::2]
            color = color_kpt[idx % len(color_kpt)] if len(color_kpt) > 0 else tuple((np.random.random(3) * 0.6 + 0.4 * 255).astype(int))

            # Draw keypoints
            for sk in sks:
                for i in range(len(sk) - 1):
                    start_point = (int(x[sk[i]]), int(y[sk[i]]))
                    end_point = (int(x[sk[i + 1]]), int(y[sk[i + 1]]))
                    cv2.line(image_np, start_point, end_point, color, 1)

            for i in range(num_kpts):
                cv2.circle(image_np, (int(x[i]), int(y[i])), 4, color, -1)

    return image_np





def load_image(cv_image):
    cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(cv_image_rgb)
    
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image_pil)
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = Config.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_unipose_output(model, image, instance_text_prompt, keypoint_text_prompt, box_threshold, iou_threshold, cpu_only=False):
    device = "cuda" if not cpu_only else "cpu"
    
    # Convert text prompts into embeddings
    instance_list = instance_text_prompt.split(',')
    ins_text_embeddings, kpt_text_embeddings = text_encoding(instance_list, keypoint_text_prompt, model.clip_model, device)

    # Prepare the target dictionary
    target = {
        "instance_text_prompt": instance_list,
        "keypoint_text_prompt": keypoint_text_prompt,
        "object_embeddings_text": ins_text_embeddings.float(),
        "kpts_embeddings_text": torch.cat([
            kpt_text_embeddings,
            torch.zeros(100 - kpt_text_embeddings.shape[0], 512, device=device)
        ], dim=0),
        "kpt_vis_text": torch.cat([
            torch.ones(kpt_text_embeddings.shape[0], device=device),
            torch.zeros(100 - kpt_text_embeddings.shape[0], device=device)
        ], dim=0)
    }

    # Ensure the model is on the correct device
    model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], [target])

    # Extract and process outputs
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]
    keypoints = outputs["pred_keypoints"][0][:, :2 * len(keypoint_text_prompt)]

    # Move to CPU for further processing
    logits = logits.cpu()
    boxes = boxes.cpu()
    keypoints = keypoints.cpu()

    # Apply the object score threshold
    max_logits = logits.max(dim=1)[0]
    keep = max_logits > box_threshold
    logits = logits[keep]
    boxes = boxes[keep]
    keypoints = keypoints[keep]

    # Apply Non-Maximum Suppression (NMS)
    keep_indices = nms(box_ops.box_cxcywh_to_xyxy(boxes), logits.max(dim=1)[0], iou_threshold=iou_threshold)
    filtered_boxes = boxes[keep_indices]
    filtered_keypoints = keypoints[keep_indices]

    return filtered_boxes, filtered_keypoints



# def run_unipose_inference(config_file, checkpoint_path, cv_image, instance_text_prompt, keypoint_text_example=None, keypoint_text_prompt="keypoints", box_threshold=0.1, iou_threshold=0.9, cpu_only=False):
#     instance_text_prompt = "person"
    
#     if instance_text_prompt in globals():
#         keypoint_dict = globals()[instance_text_prompt]
#         keypoint_text_prompt = keypoint_dict.get("keypoints")
#         keypoint_skeleton = keypoint_dict.get("skeleton")
#     else:
#         keypoint_dict = globals()["animal"]
#         keypoint_text_prompt = keypoint_dict.get("keypoints")
#         keypoint_skeleton = keypoint_dict.get("skeleton")
    
#     image_pil, image = load_image(cv_image)
#     model = load_model(config_file, checkpoint_path, cpu_only=cpu_only)

#     boxes_filt, keypoints_filt = get_unipose_output(
#         model, image, instance_text_prompt, keypoint_text_prompt, box_threshold, iou_threshold, cpu_only=cpu_only
#     )

#     pred_dict = {
#         "boxes": boxes_filt,
#         "keypoints": keypoints_filt,
#         "size": [image_pil.size[1], image_pil.size[0]]
#     }

#     output_image = plot_on_image(image_pil, pred_dict, keypoint_skeleton, keypoint_text_prompt)
#     output_image_cv = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

#     return output_image_cv


class UniPoseLiveInferencer:
    def __init__(self, config_file, checkpoint_path, cpu_only=False):
        self.device = "cpu" if cpu_only else "cuda"
        self.model = self.load_model(config_file, checkpoint_path)
        
        # Set default instance and keypoint prompts
        self.instance_text_prompt = "person"
        self.keypoint_dict = globals().get(self.instance_text_prompt, {})
        self.keypoint_text_prompt = self.keypoint_dict.get("keypoints", [])
        self.keypoint_skeleton = self.keypoint_dict.get("skeleton", [])

    def load_model(self, config_file, checkpoint_path):
        model = load_model(config_file, checkpoint_path, cpu_only=(self.device == "cpu"))
        model.to(self.device)
        return model

    def run_inference(self, cv_image, box_threshold=0.1, iou_threshold=0.9):
        # Convert the cv2 image to a PIL image
        image_pil, image = load_image(cv_image)

        # Get the keypoints and skeleton prompts
        instance_text_prompt = self.instance_text_prompt
        keypoint_text_prompt = self.keypoint_text_prompt
        keypoint_skeleton = self.keypoint_skeleton

        # Run the model to get bounding boxes and keypoints
        boxes_filt, keypoints_filt = get_unipose_output(
            self.model, image, instance_text_prompt, keypoint_text_prompt, 
            box_threshold=box_threshold, iou_threshold=iou_threshold, 
            cpu_only=(self.device == "cpu")
        )

        # Prepare prediction dictionary for visualization
        size = image_pil.size
        pred_dict = {
            "boxes": boxes_filt,
            "keypoints": keypoints_filt,
            "size": [size[1], size[0]]
        }

        # Plot keypoints on the image and return the output
        output_image = plot_on_image(image_pil, pred_dict, keypoint_skeleton, keypoint_text_prompt)
        
        # Convert output image back to cv2 format
        output_image_cv = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        
        return output_image_cv



# Example usage
if __name__ == "__main__":
    config_file = "/workspace/src/unipose/unipose/config_model/UniPose_SwinT.py"  # Path to config file
    checkpoint_path = "/workspace/src/unipose/unipose/config_model/unipose_swint.pth"  # Path to checkpoint file
    image_path = "/workspace/src/unipose/unipose_d/dev_data/climbing.jpg"  # Path to the image file
    instance_text_prompt = "person"  # Instance text prompt
    keypoint_text_example = "person"  # Keypoint text prompt (optional)
    output_dir = "/workspace/src/unipose_d/unipose_d"  # Output directory

    cv_image = cv2.imread(image_path)

    # # Call the function
    # image = run_unipose_inference(
    #     config_file = config_file, 
    #     checkpoint_path = checkpoint_path, 
    #     cv_image = cv_image, 
    #     instance_text_prompt = instance_text_prompt
    #     )
    
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # # Step 3: Plot the image using matplotlib
    # print(image_rgb)