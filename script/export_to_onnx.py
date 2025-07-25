import argparse
import os
import torch
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights
from collections import OrderedDict


class FCOSBackboneOnly(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.backbone = original_model.backbone
        self.anchor_generator = original_model.anchor_generator
        self.head = original_model.head
        self.transform = original_model.transform

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
            'num_anchors_per_level': num_anchors_per_level  # Added for completeness
        }

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', type=str, default='models',
                help='Path to save the exported model')
ap.add_argument('--height', type=int, default=374,
                help='Input image height')
ap.add_argument('--width', type=int, default=1238,
                help='Input image width')
args = vars(ap.parse_args())

# Create output directory if it doesn't exist
os.makedirs(args['output'], exist_ok=True)

# Load pretrained FCOS model
print("Loading pretrained FCOS model...")
original_model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT)
original_model.eval()

# Create backbone-only version
model = FCOSBackboneOnly(original_model)
model.eval()

# Create dummy input - note: input should be a list of tensors for proper transform handling
height, width = args['height'], args['width']
dummy_input = [torch.randn(3, height, width)]  # List of tensors, not batched tensor

print(f"Exporting model with input size: {height}x{width}")

# Export to ONNX (backbone + heads only, no NMS)
output_path = os.path.join(args['output'], f'fcos_resnet50_fpn_{height}x{width}.onnx')

torch.onnx.export(
    model,
    dummy_input,
    output_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=[
        'cls_logits',
        'bbox_regression',
        'bbox_ctrness',
        'anchors',
        'image_sizes',
        'original_image_sizes',
        'num_anchors_per_level'
    ],
    # Note: Dynamic axes might be tricky with list inputs and anchor generation
    # Consider using fixed batch size for more reliable ONNX export
    verbose=True
)

print(f"✓ Backbone model successfully exported to ONNX format as '{output_path}'")
print("✓ Model outputs raw predictions without NMS post-processing")
print("⚠ Note: You'll need to implement NMS separately in your inference pipeline")

# Test the exported model
print("\nTesting ONNX model...")
import onnx
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print("✓ ONNX model validation passed")
