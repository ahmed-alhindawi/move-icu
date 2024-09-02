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

# import sys
# sys.path.insert(1, '.')

# import argparse
# import os
# import sys
# import numpy as np
# import torch
# from PIL import Image, ImageDraw, ImageFont
# import clip
# #import transforms as T
# from .models import build_model
# from src.unipose.unipose.predefined_keypoints import *
# from .util import box_ops
# from .util.config import Config
# from .util.utils import clean_state_dict
# import matplotlib.pyplot as plt
# from matplotlib.collections import PatchCollection
# from matplotlib.patches import Polygon
# #from matplotlib import transforms
# from torchvision.ops import nms
# import torchvision.transforms as T

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
    
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(image_pil, aspect='equal')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal')
    ax.axis('off')

    color_kpt = [
        [0.00, 0.00, 0.00], [1.00, 1.00, 1.00], [1.00, 0.00, 0.00],
        [1.00, 1.00, 0.00], [0.50, 0.16, 0.16], [0.00, 0.00, 1.00],
        [0.69, 0.88, 0.90], [0.00, 1.00, 0.00], [0.63, 0.13, 0.94],
        [0.82, 0.71, 0.55], [1.00, 0.38, 0.00], [0.53, 0.15, 0.34],
        [1.00, 0.39, 0.28], [1.00, 0.00, 1.00], [0.04, 0.09, 0.27],
        [0.20, 0.63, 0.79], [0.94, 0.90, 0.55], [0.33, 0.42, 0.18],
        [0.53, 0.81, 0.92], [0.71, 0.49, 0.86], [0.25, 0.88, 0.82]
    ]
    color_box = [0.53, 0.81, 0.92]

    polygons = []
    colors = []

    for box in tgt['boxes'].cpu():
        unnormbbox = box * torch.Tensor([W, H, W, H])
        unnormbbox[:2] -= unnormbbox[2:] / 2
        bbox_x, bbox_y, bbox_w, bbox_h = unnormbbox.tolist()
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        colors.append(color_box)

    p = PatchCollection(polygons, facecolor=colors, linewidths=0, alpha=0.1)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', linestyle="--", edgecolors=colors, linewidths=1.5)
    ax.add_collection(p)

    if 'keypoints' in tgt:
        #import pdb;pdb.set_trace()
        sks = np.array(keypoint_skeleton)
        if sks.min() == 1:
            sks -= 1

        for idx, ann in enumerate(tgt['keypoints']):
            kp = np.array(ann.cpu())
            Z = kp[:num_kpts * 2] * np.array([W, H] * num_kpts)
            x = Z[0::2]
            y = Z[1::2]
            c = color_kpt[idx % len(color_kpt)] if len(color_kpt) > 0 else (np.random.random(3) * 0.6 + 0.4).tolist()

            for sk in sks:
                ax.plot(x[sk], y[sk], linewidth=1, color=c)

            for i in range(num_kpts):
                c_kpt = color_kpt[i]
                ax.plot(x[i], y[i], 'o', markersize=4, markerfacecolor=c_kpt, markeredgecolor='k', markeredgewidth=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    image_np = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    #import pdb;pdb.set_trace()
    return image_np


# Example usage
# img_cv = plot_to_cv_image(image_pil, tgt, keypoint_skeleton, keypoint_text_prompt)
# Now
# you can use img_cv as an OpenCV image






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
    instance_list = instance_text_prompt.split(',')
    device = "cuda" if not cpu_only else "cpu"
    ins_text_embeddings, kpt_text_embeddings = text_encoding(instance_list, keypoint_text_prompt, model.clip_model, device)

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

    model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], [target])

    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]
    keypoints = outputs["pred_keypoints"][0][:, :2 * len(keypoint_text_prompt)]

    logits_filt = logits.cpu()
    boxes_filt = boxes.cpu()
    keypoints_filt = keypoints.cpu()

    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]
    keypoints_filt = keypoints_filt[filt_mask]

    keep_indices = nms(box_ops.box_cxcywh_to_xyxy(boxes_filt), logits_filt.max(dim=1)[0], iou_threshold=iou_threshold)

    filtered_boxes = boxes_filt[keep_indices]
    filtered_keypoints = keypoints_filt[keep_indices]

    return filtered_boxes, filtered_keypoints




def run_unipose_inference(config_file, checkpoint_path, cv_image, instance_text_prompt, keypoint_text_example=None, keypoint_text_prompt="keypoints", box_threshold=0.1, iou_threshold=0.9, cpu_only=False):
    # Set the keypoint and skeleton information for "person"
    # keypoint_dict = globals().get("person", {})
    # keypoint_text_prompt = keypoint_dict.get("keypoints", [])
    # keypoint_skeleton = keypoint_dict.get("skeleton", [])

    instance_text_prompt = "person"

    # if keypoint_text_example in globals():
    #     keypoint_dict = globals()[keypoint_text_example]
    #     keypoint_text_prompt = keypoint_dict.get("keypoints")
    #     keypoint_skeleton = keypoint_dict.get("skeleton")
    if instance_text_prompt in globals():
        keypoint_dict = globals()[instance_text_prompt]
        keypoint_text_prompt = keypoint_dict.get("keypoints")
        keypoint_skeleton = keypoint_dict.get("skeleton")
    else:
        keypoint_dict = globals()["animal"]
        keypoint_text_prompt = keypoint_dict.get("keypoints")
        keypoint_skeleton = keypoint_dict.get("skeleton")

    image_pil, image = load_image(cv_image)
    model = load_model(config_file, checkpoint_path, cpu_only=cpu_only)

    boxes_filt, keypoints_filt = get_unipose_output(
        model, image, instance_text_prompt, keypoint_text_prompt, box_threshold, iou_threshold, cpu_only=cpu_only
    )

    pred_dict = {
        "boxes": boxes_filt,
        "keypoints": keypoints_filt,
        "size": [image_pil.size[1], image_pil.size[0]]
    }

    output_image = plot_on_image(image_pil, pred_dict, keypoint_skeleton, keypoint_text_prompt)
    output_image_cv = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    return output_image_cv




class UniPoseLiveInferencer:
    def __init__(self, config_file, checkpoint_path, cpu_only=False):
        self.device = "cuda" if not cpu_only else "cpu"
        self.model = self.load_model(config_file, checkpoint_path, cpu_only)

    def load_model(self, config_file, checkpoint_path, cpu_only):
        model = load_model(config_file, checkpoint_path, cpu_only=cpu_only)
        model = model.to(self.device)  # Ensure the model is moved to the correct device
        return model

    def run_inference(self, cv_image, box_threshold=0.1, iou_threshold=0.9):
        # Use predefined keypoint and instance text prompts
        keypoint_dict = globals().get("person", {})
        keypoint_text_prompt = keypoint_dict.get("keypoints", [])
        keypoint_skeleton = keypoint_dict.get("skeleton", [])


            # Convert the cv2 image to a PIL image
        image_pil, image = load_image(cv_image)
        iou_threshold=0.9
        instance_text_prompt = "person"

        if instance_text_prompt in globals():
            keypoint_dict = globals()[instance_text_prompt]
            keypoint_text_prompt = keypoint_dict.get("keypoints")
            keypoint_skeleton = keypoint_dict.get("skeleton")
        else:
            keypoint_dict = globals()["animal"]
            keypoint_text_prompt = keypoint_dict.get("keypoints")
            keypoint_skeleton = keypoint_dict.get("skeleton")

        # Run the model to get bounding boxes and keypoints
        boxes_filt, keypoints_filt = get_unipose_output(
            self.model, image, instance_text_prompt, keypoint_text_prompt, box_threshold=0.1, iou_threshold=0.9, cpu_only=(self.device == "cpu")
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

    # Call the function
    image = run_unipose_inference(
        config_file = config_file, 
        checkpoint_path = checkpoint_path, 
        cv_image = cv_image, 
        instance_text_prompt = instance_text_prompt
        )
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 3: Plot the image using matplotlib
    print(image_rgb)