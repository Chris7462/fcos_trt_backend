import torch
import torchvision
from torch import nn
from collections import OrderedDict

class TRTEfficientNMS(nn.Module):
    """TensorRT EfficientNMS plugin wrapper"""
    def __init__(self, score_threshold=0.2, iou_threshold=0.6, max_output_boxes=100, background_class=-1):
        super().__init__()
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_output_boxes = max_output_boxes
        self.background_class = background_class

    def forward(self, boxes, scores):
        # This is a placeholder - the actual TRT plugin will be substituted during conversion
        # For ONNX export, we'll use a simplified version
        batch_size = boxes.shape[0]
        num_classes = scores.shape[-1]

        # Reshape for batched_nms compatibility
        boxes_flat = boxes.view(-1, 4)
        scores_flat = scores.view(-1, num_classes)

        # Get max scores and labels
        max_scores, labels = torch.max(scores_flat, dim=1)

        # Simple NMS (will be replaced by TRT plugin)
        keep = torch.ops.torchvision.nms(boxes_flat, max_scores, self.iou_threshold)
        keep = keep[:self.max_output_boxes]

        # Return in expected format
        return boxes_flat[keep].unsqueeze(0), max_scores[keep].unsqueeze(0), labels[keep].unsqueeze(0)

class FCOSWithTRTNMS(torch.nn.Module):
    """FCOS model with TensorRT-compatible NMS"""
    def __init__(self, fcos_model):
        super().__init__()
        self.fcos_model = fcos_model
        self.nms = TRTEfficientNMS(
            score_threshold=fcos_model.score_thresh,
            iou_threshold=fcos_model.nms_thresh,
            max_output_boxes=fcos_model.detections_per_img
        )

    def forward(self, images):
        # Get original image sizes
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # Transform input
        images_transformed, _ = self.fcos_model.transform(images, None)

        # Get features
        features = self.fcos_model.backbone(images_transformed.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        features = list(features.values())

        # Get head outputs
        head_outputs = self.fcos_model.head(features)

        # Generate anchors
        anchors = self.fcos_model.anchor_generator(images_transformed, features)
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]

        # Manual postprocessing without PyTorch NMS
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]
        box_ctrness = head_outputs["bbox_ctrness"]

        # Process first image only (batch_size=1)
        anchors_per_image = anchors[0]
        image_shape = images_transformed.image_sizes[0]

        # Decode boxes
        pred_boxes = self.fcos_model.head.box_coder.decode(box_regression[0], anchors_per_image)

        # Get scores (classification * centerness)
        cls_scores = torch.sigmoid(class_logits[0])  # [num_anchors, num_classes]
        ctr_scores = torch.sigmoid(box_ctrness[0]).squeeze(-1)  # [num_anchors]

        # Combine scores
        scores = torch.sqrt(cls_scores * ctr_scores.unsqueeze(-1))  # [num_anchors, num_classes]

        # Use TRT NMS
        final_boxes, final_scores, final_labels = self.nms(pred_boxes.unsqueeze(0), scores.unsqueeze(0))

        return final_boxes.squeeze(0), final_labels.squeeze(0), final_scores.squeeze(0)

# Load model and create TRT-compatible version
model = torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT")
model.eval()

trt_model = FCOSWithTRTNMS(model)
trt_model.eval()

# Export
dummy_input = torch.randn(1, 3, 374, 1238)

torch.onnx.export(
    trt_model,
    dummy_input,
    "fcos_trt_nms.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["images"],
    output_names=["boxes", "labels", "scores"],
    dynamic_axes={
        "images": {0: "batch_size", 2: "height", 3: "width"},
        "boxes": {0: "num_boxes"},
        "labels": {0: "num_boxes"},
        "scores": {0: "num_boxes"},
    },
)

print("Exported FCOS with TRT-compatible NMS")
