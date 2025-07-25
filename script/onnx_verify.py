import torch
import torchvision
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
from torchvision.ops import boxes as box_ops
from collections import OrderedDict
import numpy as np
from typing import List, Dict, Tuple, Any
from torchvision.models.detection import _utils as det_utils
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt


class FCOSBackboneOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained FCOS model
        print("Loading pretrained FCOS model...")
        self.model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()
        self.backbone = self.model.backbone
        self.anchor_generator = self.model.anchor_generator
        self.head = self.model.head
        self.transform = self.model.transform

    def forward(self, images):
        # Store original image sizes before transformation
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

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
            'anchors': anchors,
            'image_sizes': images.image_sizes,
            'original_image_sizes': original_image_sizes,
            'num_anchors_per_level': num_anchors_per_level
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
        anchors: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
        num_anchors_per_level: List[int]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Postprocess the raw outputs from FCOSBackboneOnly to get final detections
        """
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

# COCO class labels (from your plotting script)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def plot_detections(image_path, detections, title, confidence_threshold=0.6):
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

        detection_count = 0
        for box, score, label in zip(boxes, scores, labels):
            if score >= confidence_threshold:
                detection_count += 1
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image_for_plot, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Draw label and score
                class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
                label_text = f'{class_name}: {score:.2f}'
                cv2.putText(image_for_plot, label_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        print(f"   {title}: {detection_count} detections above {confidence_threshold} confidence")
        return cv2.cvtColor(image_for_plot, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"Error plotting detections: {e}")
        return None


def visualize_comparison(image_path, original_results, custom_results, confidence_threshold=0.6):
    """Create side-by-side comparison of detection results"""

    print(f"\n=== Visualization Comparison (confidence > {confidence_threshold}) ===")

    # Plot original results
    original_plot = plot_detections(image_path, original_results[0], "Original FCOS", confidence_threshold)

    # Plot custom results
    custom_plot = plot_detections(image_path, custom_results[0], "Custom FCOSBackboneOnly", confidence_threshold)

    if original_plot is not None and custom_plot is not None:
        # Create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        ax1.imshow(original_plot)
        ax1.set_title('Original FCOS Model', fontsize=14, fontweight='bold')
        ax1.axis('off')

        ax2.imshow(custom_plot)
        ax2.set_title('Custom FCOSBackboneOnly + PostProcessor', fontsize=14, fontweight='bold')
        ax2.axis('off')

        plt.tight_layout()
        plt.suptitle(f'FCOS Detection Comparison (confidence > {confidence_threshold})', 
                    fontsize=16, fontweight='bold', y=0.98)

        # Save the comparison plot
        try:
            plt.savefig('fcos_comparison.png', dpi=150, bbox_inches='tight')
            print("   ✅ Comparison plot saved as 'fcos_comparison.png'")
        except Exception as e:
            print(f"   ⚠️  Could not save plot: {e}")

        plt.show()
    else:
        print("   ❌ Could not create visualization - image loading failed")


def compare_models():
    """Compare FCOSBackboneOnly + PostProcessor vs Original FCOS"""

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)


    # Load real test image
    test_image_path = "fcos_torch_backend/script/image_000.png"

    try:
        from PIL import Image
        import torchvision.transforms as transforms

        # Load and preprocess the image
        pil_image = Image.open(test_image_path).convert('RGB')
        transform = transforms.ToTensor()
        test_image_tensor = transform(pil_image)
        test_images = [test_image_tensor]

        print(f"Loaded image: {test_image_path}")
        print(f"Image shape: {test_image_tensor.shape}")

    except Exception as e:
        print(f"Error loading image {test_image_path}: {e}")
        print("Falling back to random test image...")
        test_images = [torch.rand(3, 400, 600)]

    print("=== Model Comparison Test ===")

    # 1. Original FCOS model
    print("\n1. Testing Original FCOS Model...")
    original_model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT)
    original_model.eval()

    with torch.no_grad():
        original_results = original_model(test_images)

    print(f"   Original model - Number of detections: {[len(r['boxes']) for r in original_results]}")

    # 2. FCOSBackboneOnly + PostProcessor
    print("\n2. Testing FCOSBackboneOnly + PostProcessor...")
    backbone_model = FCOSBackboneOnly()
    post_processor = FCOSPostProcessor(
        score_thresh=original_model.score_thresh,
        nms_thresh=original_model.nms_thresh,
        detections_per_img=original_model.detections_per_img,
        topk_candidates=original_model.topk_candidates
    )

    with torch.no_grad():
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
        )

        # Apply transform postprocess to get final coordinates
        custom_results = original_model.transform.postprocess(
            custom_results,
            raw_outputs['image_sizes'],
            raw_outputs['original_image_sizes']
        )

    print(f"   Custom model - Number of detections: {[len(r['boxes']) for r in custom_results]}")

    # 3. Compare results
    print("\n3. Comparing Results...")

    total_differences = []

    for i, (orig, custom) in enumerate(zip(original_results, custom_results)):
        print(f"\n   Image {i+1}:")
        print(f"   Original detections: {len(orig['boxes'])}")
        print(f"   Custom detections:   {len(custom['boxes'])}")

        if len(orig['boxes']) == 0 and len(custom['boxes']) == 0:
            print("   ✓ Both models found no detections")
            continue

        # Compare top detections (sort by score)
        max_compare = min(len(orig['boxes']), len(custom['boxes']), 5)

        if max_compare > 0:
            # Sort by scores (descending)
            orig_sorted_idx = torch.argsort(orig['scores'], descending=True)
            custom_sorted_idx = torch.argsort(custom['scores'], descending=True)

            orig_top_boxes = orig['boxes'][orig_sorted_idx[:max_compare]]
            orig_top_scores = orig['scores'][orig_sorted_idx[:max_compare]]
            orig_top_labels = orig['labels'][orig_sorted_idx[:max_compare]]

            custom_top_boxes = custom['boxes'][custom_sorted_idx[:max_compare]]
            custom_top_scores = custom['scores'][custom_sorted_idx[:max_compare]]
            custom_top_labels = custom['labels'][custom_sorted_idx[:max_compare]]

            # Compare top predictions
            box_diff = torch.abs(orig_top_boxes - custom_top_boxes).max().item()
            score_diff = torch.abs(orig_top_scores - custom_top_scores).max().item()
            label_match = torch.equal(orig_top_labels, custom_top_labels)

            print(f"   Top {max_compare} detections comparison:")
            print(f"   Max box coordinate difference: {box_diff:.6f}")
            print(f"   Max score difference: {score_diff:.6f}")
            print(f"   Labels match: {label_match}")

            total_differences.extend([box_diff, score_diff])

            # Show detailed comparison for top detection
            if max_compare > 0:
                print(f"   \n   Top detection details:")
                print(f"   Original - Box: {orig_top_boxes[0]}, Score: {orig_top_scores[0]:.4f}, Label: {orig_top_labels[0]}")
                print(f"   Custom   - Box: {custom_top_boxes[0]}, Score: {custom_top_scores[0]:.4f}, Label: {custom_top_labels[0]}")

    # Summary
    print(f"\n=== Summary ===")
    if total_differences:
        max_diff = max(total_differences)
        avg_diff = np.mean(total_differences)
        print(f"Maximum difference: {max_diff:.8f}")
        print(f"Average difference: {avg_diff:.8f}")

        if max_diff < 1e-5:
            print("✅ EXCELLENT: Models produce nearly identical results!")
        elif max_diff < 1e-3:
            print("✅ GOOD: Models produce very similar results (small numerical differences)")
        elif max_diff < 0.1:
            print("⚠️  ACCEPTABLE: Models produce similar results (some differences present)")
        else:
            print("❌ SIGNIFICANT: Models produce different results - investigation needed")
    else:
        print("No detections found by either model for comparison")

    return original_results, custom_results


def detailed_intermediate_comparison():
    """Compare intermediate outputs to identify where differences might occur"""

    print("\n=== Detailed Intermediate Comparison ===")

    torch.manual_seed(42)

    # Load the same test image
    test_image_path = "fcos_torch_backend/script/image_000.png"

    try:
        from PIL import Image
        import torchvision.transforms as transforms

        # Load and preprocess the image
        pil_image = Image.open(test_image_path).convert('RGB')
        transform = transforms.ToTensor()
        test_image_tensor = transform(pil_image)
        test_image = [test_image_tensor]

        print(f"Using test image: {test_image_path}")
        print(f"Image shape: {test_image_tensor.shape}")

    except Exception as e:
        print(f"Error loading image {test_image_path}: {e}")
        print("Falling back to random test image...")
        test_image = [torch.rand(3, 300, 400)]

    # Original model
    original_model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT)
    original_model.eval()

    # Custom model
    backbone_model = FCOSBackboneOnly()

    with torch.no_grad():
        # Get intermediate outputs from both models
        print("\n1. Comparing backbone features...")

        # For original model, we need to manually extract intermediate results
        original_images, _ = original_model.transform(test_image, None)
        original_features = original_model.backbone(original_images.tensors)
        if isinstance(original_features, torch.Tensor):
            original_features = OrderedDict([("0", original_features)])
        original_features_list = list(original_features.values())

        # Get features from custom model
        custom_outputs = backbone_model(test_image)

        # Compare transformed images (should be identical)
        print(f"   Image tensor shapes match: {original_images.tensors.shape}")
        print(f"   Image sizes: {original_images.image_sizes}")

        # Compare anchor generation
        original_anchors = original_model.anchor_generator(original_images, original_features_list)
        print(f"   Original anchors type: {type(original_anchors[0])}")
        print(f"   Custom anchors type: {type(custom_outputs['anchors'][0])}")

        # Handle different anchor formats
        if isinstance(original_anchors[0], list):
            print(f"   Anchors shapes - Original: {[a.shape for a in original_anchors[0]]}")
            original_anchors_concat = torch.cat(original_anchors[0])
        else:
            print(f"   Anchors shape - Original: {original_anchors[0].shape}")
            original_anchors_concat = original_anchors[0]

        if isinstance(custom_outputs['anchors'][0], list):
            print(f"   Anchors shapes - Custom: {[a.shape for a in custom_outputs['anchors'][0]]}")
            custom_anchors_concat = torch.cat(custom_outputs['anchors'][0])
        else:
            print(f"   Anchors shape - Custom: {custom_outputs['anchors'][0].shape}")
            custom_anchors_concat = custom_outputs['anchors'][0]

        anchor_diff = torch.abs(original_anchors_concat - custom_anchors_concat).max()
        print(f"   Max anchor difference: {anchor_diff:.8f}")

        # Compare head outputs
        original_head_outputs = original_model.head(original_features_list)

        cls_diff = torch.abs(original_head_outputs['cls_logits'] - custom_outputs['cls_logits']).max()
        bbox_diff = torch.abs(original_head_outputs['bbox_regression'] - custom_outputs['bbox_regression']).max()  
        ctr_diff = torch.abs(original_head_outputs['bbox_ctrness'] - custom_outputs['bbox_ctrness']).max()

        print(f"   Max cls_logits difference: {cls_diff:.8f}")
        print(f"   Max bbox_regression difference: {bbox_diff:.8f}")
        print(f"   Max bbox_ctrness difference: {ctr_diff:.8f}")

        if max(anchor_diff, cls_diff, bbox_diff, ctr_diff) < 1e-6:
            print("   ✅ All intermediate outputs are identical!")
        else:
            print("   ⚠️  Some small differences in intermediate outputs")


if __name__ == "__main__":
    # Run the comparison
    original_results, custom_results = compare_models()

    # Run detailed intermediate comparison
    detailed_intermediate_comparison()

    test_image_path = "fcos_torch_backend/script/image_000.png"

    # Create visual comparison
    visualize_comparison(test_image_path, original_results, custom_results, confidence_threshold=0.6)

    print("\n=== Test Complete ===")
    print("If the models produce nearly identical results, your FCOSBackboneOnly")
    print("implementation is correct and ready for ONNX export!")
    print("Check 'fcos_comparison.png' for the visual comparison of detection results.")
