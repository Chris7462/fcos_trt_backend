#!/usr/bin/env python3
"""Script to verify FCOS postprocessing implementation by comparing with original model."""

import torch
from torch.nn import Module
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights
from torchvision.ops import boxes as box_ops
from collections import OrderedDict
import numpy as np
from typing import List, Dict, Tuple, Any
from torchvision.models.detection import _utils as det_utils
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import argparse
import os


# Correct COCO class mapping using actual category IDs (with gaps)
COCO_INSTANCE_CATEGORY_NAMES = {
    0: '__background__',  # Background class (though typically not returned in detections)
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}
# IDs 12, 26, 29, 30, 45, 66, 68, 69, 71, 83 are not used (they were categories in early drafts of COCO but removed).

def load_and_preprocess_image(image_path):
    """Load and preprocess image to tensor format."""
    print(f"Loading image: {image_path}")

    try:
        # Load image
        pil_image = Image.open(image_path).convert('RGB')
        original_size = pil_image.size  # (width, height)
        print(f"Original image size: {original_size}")

        # Convert to tensor
        transform = transforms.ToTensor()
        image_tensor = transform(pil_image)

        print(f"Image tensor shape: {image_tensor.shape}")
        return [image_tensor], original_size

    except Exception as e:
        print(f"Error loading image: {e}")
        raise


class FCOSBackbone(Module):
    """Wrapper to extract only the Backbone from FCOS model."""
    def __init__(self):
        super().__init__()
        # Load pretrained FCOS model
        print('Loading pretrained FCOS model...')
        self.model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()
        self.backbone = self.model.backbone
        self.anchor_generator = self.model.anchor_generator
        self.head = self.model.head
        self.transform = self.model.transform

    def forward(self, images):
        # Apply transforms (normalization, resizing)
        images, _ = self.transform(images, None)

        # Extract features using backbone
        features = self.backbone(images.tensors)

        # Handle case where backbone returns a single tensor (convert to OrderedDict)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # Convert to list as expected by head and anchor_generator
        features_list = list(features.values())

        # Generate anchors - pass both images and features
        anchors = self.anchor_generator(images, features_list)

        # Get predictions from head
        head_outputs = self.head(features_list)

        # Calculate num_anchors_per_level for consistency with original forward
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features_list]

        # Return raw outputs without NMS post-processing
        return {
            'cls_logits': head_outputs['cls_logits'],
            'bbox_regression': head_outputs['bbox_regression'],
            'bbox_ctrness': head_outputs['bbox_ctrness'],
            'anchors': anchors[0],
            'image_sizes': torch.tensor(images.image_sizes[0]),
            'num_anchors_per_level': torch.tensor(num_anchors_per_level)
        }

class FCOSPostProcessor:
    """Standalone post-processor that mimics the original FCOS postprocess_detections method"""

    def __init__(self, score_thresh=0.2, nms_thresh=0.6, detections_per_img=100, topk_candidates=1000):
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.box_coder = det_utils.BoxLinearCoder(normalize_by_size=True)

    def postprocess_detections(
        self,
        cls_logits: torch.Tensor,
        bbox_regression: torch.Tensor,
        bbox_ctrness: torch.Tensor,
        anchors: torch.Tensor,  # Now single tensor instead of list
        image_sizes: torch.Tensor,  # Now single tensor instead of list
        num_anchors_per_level: torch.Tensor  # Now tensor instead of list
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Postprocess the raw outputs from FCOSBackbone to get final detections
        """
        # Convert tensors back to expected formats
        num_anchors_per_level = num_anchors_per_level.tolist()
        image_sizes = [tuple(image_sizes.tolist())]  # Convert back to list of tuples
        anchors = [anchors]  # Convert back to list

        num_images = len(image_sizes)
        detections: List[Dict[str, torch.Tensor]] = []

        # Split outputs per level (similar to original FCOS)
        split_cls_logits = list(cls_logits.split(num_anchors_per_level, dim=1))
        split_bbox_regression = list(bbox_regression.split(num_anchors_per_level, dim=1))
        split_bbox_ctrness = list(bbox_ctrness.split(num_anchors_per_level, dim=1))
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

        for index in range(num_images):
            # Get per-image outputs for each level
            cls_logits_per_image = [cl[index] for cl in split_cls_logits]
            bbox_regression_per_image = [br[index] for br in split_bbox_regression]
            bbox_ctrness_per_image = [bc[index] for bc in split_bbox_ctrness]
            anchors_per_image = split_anchors[index]
            image_shape = image_sizes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            # Process each pyramid level
            for (cls_logits_per_level, bbox_regression_per_level,
                 bbox_ctrness_per_level, anchors_per_level) in zip(
                cls_logits_per_image, bbox_regression_per_image,
                bbox_ctrness_per_image, anchors_per_image
            ):
                num_classes = cls_logits_per_level.shape[-1]

                # Compute scores: sqrt(classification_score * centerness_score)
                scores_per_level = torch.sqrt(
                    torch.sigmoid(cls_logits_per_level) * torch.sigmoid(bbox_ctrness_per_level)
                ).flatten()

                # Filter by score threshold
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # Keep only topk scoring predictions
                num_topk = det_utils._topk_min(topk_idxs, self.topk_candidates, 0)
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                # Convert flat indices back to anchor and class indices
                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                # Decode bounding boxes
                boxes_per_level = self.box_coder.decode(
                    bbox_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            # Concatenate results from all levels
            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # Apply NMS
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]

            detections.append({
                "boxes": image_boxes[keep],
                "scores": image_scores[keep],
                "labels": image_labels[keep],
            })

        return detections


if __name__ == '__main__':
    args = argparse.Namespace()
    args.image_path = f'./fcos_trt_backend/test/image_000.png'
    args.confidence = 0.5
    args.tolerance = 1e-4

    image_path = args.image_path
    confidence_threshold = args.confidence
    tolerance = args.tolerance

    # Load and preprocess image
    test_images, original_size = load_and_preprocess_image(image_path)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 2. FCOSBackbone + PostProcessor
    backbone_model = FCOSBackbone()
    post_processor = FCOSPostProcessor(
        score_thresh=0.2,
        nms_thresh=0.6,
        detections_per_img=100,
        topk_candidates=1000)

    torch.no_grad()
    # Get raw outputs from backbone
    raw_outputs = backbone_model(test_images)

    # Apply post-processing
    custom_results = post_processor.postprocess_detections(
        cls_logits=raw_outputs['cls_logits'],
        bbox_regression=raw_outputs['bbox_regression'],
        bbox_ctrness=raw_outputs['bbox_ctrness'],
        anchors=raw_outputs['anchors'],
        image_sizes=raw_outputs['image_sizes'],
        num_anchors_per_level=raw_outputs['num_anchors_per_level']
    )[0]

    # debuging
    cls_logits=raw_outputs['cls_logits']
    bbox_regression=raw_outputs['bbox_regression']
    bbox_ctrness=raw_outputs['bbox_ctrness']
    anchors=raw_outputs['anchors']
    image_sizes=raw_outputs['image_sizes']
    num_anchors_per_level=raw_outputs['num_anchors_per_level']

    score_thresh=0.2
    nms_thresh=0.6
    detections_per_img=100
    topk_candidates=1000
    box_coder = det_utils.BoxLinearCoder(normalize_by_size=True)

    # Convert tensors back to expected formats
    num_anchors_per_level = num_anchors_per_level.tolist()
    image_sizes = [tuple(image_sizes.tolist())]  # Convert back to list of tuples
    anchors = [anchors]  # Convert back to list

    num_images = len(image_sizes)
    detections: List[Dict[str, torch.Tensor]] = []

    # Split outputs per level (similar to original FCOS)
    split_cls_logits = list(cls_logits.split(num_anchors_per_level, dim=1))
    split_bbox_regression = list(bbox_regression.split(num_anchors_per_level, dim=1))
    split_bbox_ctrness = list(bbox_ctrness.split(num_anchors_per_level, dim=1))
    split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

    for index in range(num_images):
        # Get per-image outputs for each level
        cls_logits_per_image = [cl[index] for cl in split_cls_logits]
        bbox_regression_per_image = [br[index] for br in split_bbox_regression]
        bbox_ctrness_per_image = [bc[index] for bc in split_bbox_ctrness]
        anchors_per_image = split_anchors[index]
        image_shape = image_sizes[index]

        image_boxes = []
        image_scores = []
        image_labels = []

        # Process each pyramid level
        for (cls_logits_per_level, bbox_regression_per_level,
            bbox_ctrness_per_level, anchors_per_level) in zip(
            cls_logits_per_image, bbox_regression_per_image,
            bbox_ctrness_per_image, anchors_per_image
        ):
            #cls_logits_per_level = cls_logits_per_image[0]
            #bbox_regression_per_level = bbox_regression_per_image[0]
            #bbox_ctrness_per_level = bbox_ctrness_per_image[0]
            #anchors_per_level = anchors_per_image[0]

            num_classes = cls_logits_per_level.shape[-1]

            # Compute scores: sqrt(classification_score * centerness_score)
            scores_per_level = torch.sqrt(
                torch.sigmoid(cls_logits_per_level) * torch.sigmoid(bbox_ctrness_per_level)
            ).flatten()

            # Filter by score threshold
            keep_idxs = scores_per_level > score_thresh
            scores_per_level = scores_per_level[keep_idxs]
            topk_idxs = torch.where(keep_idxs)[0]

            # ===== Debug info =====
            if len(scores_per_level) != 0:
                print("score size per level = ", len(scores_per_level))
                print(f"score[20] = {scores_per_level[19]:.6f}")

            # Keep only topk scoring predictions
            num_topk = det_utils._topk_min(topk_idxs, topk_candidates, 0)
            scores_per_level, idxs = scores_per_level.topk(num_topk)
            topk_idxs = topk_idxs[idxs]

            # Convert flat indices back to anchor and class indices
            anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
            labels_per_level = topk_idxs % num_classes

            # Decode bounding boxes
            print("bbox_regression_per_level[:20] = ", bbox_regression_per_level[:20])
            print("anchors_per_level[:20] = ", anchors_per_level[:20])
            print("anchor_idxs[:20] = ", anchor_idxs[:20])
            boxes_per_level = box_coder.decode(
                bbox_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
            )
            print("boxes_per_level = ", boxes_per_level[:20])
            print("scores_per_level = ", scores_per_level[:20])
            print("labels_per_level = ", labels_per_level[:20])

            boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

            image_boxes.append(boxes_per_level)
            image_scores.append(scores_per_level)
            image_labels.append(labels_per_level)

        # Concatenate results from all levels
        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)

#       # Apply NMS
#       keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
#       keep = keep[:self.detections_per_img]

#       detections.append({
#           "boxes": image_boxes[keep],
#           "scores": image_scores[keep],
#           "labels": image_labels[keep],
#       })
