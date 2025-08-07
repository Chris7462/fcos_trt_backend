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


def print_tensor_stats(tensor, name, print_first_n=20):
    """Print statistics and first N values of a tensor."""
    flat_tensor = tensor.flatten()

    print(f"{name} (first {print_first_n} values):")
    first_values = flat_tensor[:print_first_n].detach().cpu().numpy()

    # Handle integer vs float tensors for formatting
    if tensor.dtype in [torch.int32, torch.int64, torch.long]:
        values_str = ' '.join([f"{val}" for val in first_values])
    else:
        values_str = ' '.join([f"{val:.5f}" for val in first_values])

    print(f"  {values_str}")
    print(f"  ... (total size: {flat_tensor.numel()})")

    print(f"{name} Statistics:")
    if tensor.dtype in [torch.int32, torch.int64, torch.long]:
        print(f"  Min: {flat_tensor.min()}")
        print(f"  Max: {flat_tensor.max()}")
        print(f"  Mean: {flat_tensor.float().mean():.5f}")
    else:
        print(f"  Min: {flat_tensor.min():.5f}")
        print(f"  Max: {flat_tensor.max():.5f}")
        print(f"  Mean: {flat_tensor.mean():.5f}")
    print(f"  Shape: {tensor.shape}")
    print()


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


def plot_detections(image_path, detections, title, confidence_threshold=0.5):
    """Plot detection results on the image"""
    try:
        # Load image using OpenCV
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Could not load image: {image_path}")
            return None

        image_for_plot = image_bgr.copy()

        # Draw predictions with confidence > threshold
        boxes = detections['boxes']
        scores = detections['scores']
        labels = detections['labels']

        print(f"\n=== {title} Detection Results ===")
        detection_count = 0

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if score >= confidence_threshold:
                detection_count += 1
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image_for_plot, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Get class name using dictionary lookup
                label_id = label.item()
                class_name = COCO_INSTANCE_CATEGORY_NAMES.get(label_id, f'unknown_class_{label_id}')

                # Draw label and score
                label_text = f'{class_name}: {score:.3f}'
                cv2.putText(image_for_plot, label_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                print(f"Detection {detection_count}: {class_name} (ID: {label_id}) - Confidence: {score:.3f}")
                print(f"  Box: [{x1}, {y1}, {x2}, {y2}]")

        print(f"Total detections above {confidence_threshold} confidence: {detection_count}")
        return cv2.cvtColor(image_for_plot, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"Error plotting detections: {e}")
        return None


def compare_detection_results(original_results, custom_results, tolerance=1e-4):
    """Compare detection results between original and custom models."""
    print(f"\n=== Comparing Detection Results (tolerance: {tolerance}) ===")

    all_match = True

    for i, (orig, custom) in enumerate(zip(original_results, custom_results)):
        print(f"\nImage {i+1} Comparison:")
        print(f"  Original detections: {len(orig['boxes'])}")
        print(f"  Custom detections:   {len(custom['boxes'])}")

        if len(orig['boxes']) != len(custom['boxes']):
            print(f"  ✗ Number of detections differ!")
            all_match = False
            continue

        if len(orig['boxes']) == 0:
            print(f"  ✓ Both models found no detections")
            continue

        # Compare sorted results (by score, descending)
        orig_sorted_idx = torch.argsort(orig['scores'], descending=True)
        custom_sorted_idx = torch.argsort(custom['scores'], descending=True)

        orig_boxes_sorted = orig['boxes'][orig_sorted_idx]
        orig_scores_sorted = orig['scores'][orig_sorted_idx]
        orig_labels_sorted = orig['labels'][orig_sorted_idx]

        custom_boxes_sorted = custom['boxes'][custom_sorted_idx]
        custom_scores_sorted = custom['scores'][custom_sorted_idx]
        custom_labels_sorted = custom['labels'][custom_sorted_idx]

        # Check if boxes match
        boxes_match = torch.allclose(orig_boxes_sorted, custom_boxes_sorted, atol=tolerance, rtol=tolerance)
        scores_match = torch.allclose(orig_scores_sorted, custom_scores_sorted, atol=tolerance, rtol=tolerance)
        labels_match = torch.equal(orig_labels_sorted, custom_labels_sorted)

        print(f"  Boxes match: {boxes_match}")
        print(f"  Scores match: {scores_match}")
        print(f"  Labels match: {labels_match}")

        if not (boxes_match and scores_match and labels_match):
            all_match = False

            # Print difference statistics
            if not boxes_match:
                box_diff = torch.abs(orig_boxes_sorted - custom_boxes_sorted)
                print(f"    Max box difference: {box_diff.max():.8f}")
                print(f"    Mean box difference: {box_diff.mean():.8f}")

            if not scores_match:
                score_diff = torch.abs(orig_scores_sorted - custom_scores_sorted)
                print(f"    Max score difference: {score_diff.max():.8f}")
                print(f"    Mean score difference: {score_diff.mean():.8f}")

            # Show first few detections for comparison
            max_show = min(3, len(orig_boxes_sorted))
            for j in range(max_show):
                print(f"    Detection {j+1}:")
                print(f"      Original - Box: {orig_boxes_sorted[j]}, Score: {orig_scores_sorted[j]:.6f}, Label: {orig_labels_sorted[j]}")
                print(f"      Custom   - Box: {custom_boxes_sorted[j]}, Score: {custom_scores_sorted[j]:.6f}, Label: {custom_labels_sorted[j]}")

    if all_match:
        print("\n🎉 SUCCESS: All detection results match between models!")
    else:
        print("\n❌ DIFFERENCES: Detection results differ between models!")

    return all_match


def visualize_comparison(image_path, original_results, custom_results, confidence_threshold=0.5):
    """Create side-by-side comparison of detection results"""
    print(f"\n=== Creating Visual Comparison (confidence > {confidence_threshold}) ===")

    # Plot original results
    original_plot = plot_detections(image_path, original_results[0], "Original FCOS", confidence_threshold)

    # Plot custom results
    custom_plot = plot_detections(image_path, custom_results[0], "Custom FCOSBackbone", confidence_threshold)

    if original_plot is not None and custom_plot is not None:
        # Create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        ax1.imshow(original_plot)
        ax1.set_title('Original FCOS Model', fontsize=14, fontweight='bold')
        ax1.axis('off')

        ax2.imshow(custom_plot)
        ax2.set_title('Custom FCOSBackbone + PostProcessor', fontsize=14, fontweight='bold')
        ax2.axis('off')

        plt.tight_layout()
        plt.suptitle(f'FCOS Detection Comparison (confidence > {confidence_threshold})',
                    fontsize=16, fontweight='bold', y=0.98)

        # Save the comparison plot
        try:
            plt.savefig('fcos_postprocess_comparison.png', dpi=150, bbox_inches='tight')
            print("✓ Comparison plot saved as 'fcos_postprocess_comparison.png'")
        except Exception as e:
            print(f"⚠️  Could not save plot: {e}")

        plt.show()
    else:
        print("❌ Could not create visualization - image loading failed")


def run_inference_comparison(image_path, confidence_threshold=0.5, tolerance=1e-4):
    """Run inference comparison between original FCOS and custom FCOSBackbone + PostProcessor."""

    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}")
        raise FileNotFoundError(f"Test image not found at {image_path}")

    print(f"=== FCOS Postprocessing Verification ===")
    print(f"Test image: {image_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Comparison tolerance: {tolerance}")

    # Load and preprocess image
    test_images, original_size = load_and_preprocess_image(image_path)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("\n=== Running Original FCOS Model ===")
    # 1. Original FCOS model
    original_model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT)
    original_model.eval()

    with torch.no_grad():
        original_results = original_model(test_images)

    print("✓ Original FCOS inference completed!")

    print("\n=== Running Custom FCOSBackbone + PostProcessor ===")
    # 2. FCOSBackbone + PostProcessor
    backbone_model = FCOSBackbone()
    post_processor = FCOSPostProcessor(
        score_thresh=original_model.score_thresh,
        nms_thresh=original_model.nms_thresh,
        detections_per_img=original_model.detections_per_img,
        topk_candidates=original_model.topk_candidates
    )

    with torch.no_grad():
        # Get raw outputs from backbone
        raw_outputs = backbone_model(test_images)

        print("Raw outputs from FCOSBackbone:")
        print_tensor_stats(raw_outputs['cls_logits'], "CLS_LOGITS")
        print_tensor_stats(raw_outputs['bbox_regression'], "BBOX_REGRESSION")
        print_tensor_stats(raw_outputs['bbox_ctrness'], "BBOX_CTRNESS")

        # Apply post-processing
        custom_results = post_processor.postprocess_detections(
            cls_logits=raw_outputs['cls_logits'],
            bbox_regression=raw_outputs['bbox_regression'],
            bbox_ctrness=raw_outputs['bbox_ctrness'],
            anchors=raw_outputs['anchors'],
            image_sizes=raw_outputs['image_sizes'],
            num_anchors_per_level=raw_outputs['num_anchors_per_level']
        )

        # Get original image sizes for transform.postprocess
        original_image_sizes = []
        for img in test_images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # Create image_sizes list from tensor
        image_sizes_list = [tuple(raw_outputs['image_sizes'].tolist())]

        # CRITICAL: Apply transform postprocess to get final coordinates
        custom_results = original_model.transform.postprocess(
            custom_results,
            image_sizes_list,
            original_image_sizes
        )

    print("✓ Custom FCOSBackbone + PostProcessor inference completed!")

    # 3. Compare detection results
    results_match = compare_detection_results(original_results, custom_results, tolerance)

    # 4. Create visual comparison
    visualize_comparison(image_path, original_results, custom_results, confidence_threshold)

    # 5. Summary
    print(f"\n=== Final Summary ===")
    if results_match:
        print("🎉 PERFECT: Custom postprocessing implementation matches original FCOS!")
        print("Your FCOSBackbone + FCOSPostProcessor is working correctly.")
    else:
        print("❌ MISMATCH: Custom postprocessing differs from original FCOS.")
        print("Check the postprocessing implementation for potential issues.")

    print(f"\nCheck 'fcos_postprocess_comparison.png' for visual comparison.")

    return original_results, custom_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify FCOS postprocessing implementation')
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to test image')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold for visualization')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Tolerance for result comparison')

    args = parser.parse_args()

    # Run the inference comparison
    original_results, custom_results = run_inference_comparison(
        args.image_path,
        args.confidence,
        args.tolerance
    )

    print("\n=== Verification Complete ===")
    print("If results match, your postprocessing implementation is correct!")
