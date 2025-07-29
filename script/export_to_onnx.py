import argparse
import cv2
import os
import torch
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor
from collections import OrderedDict


class FCOSBackbone(torch.nn.Module):
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
        # Apply transforms (normalization, resizing)
        images, _ = self.transform(images, None)

        # Extract features using backbone
        features = self.backbone(images.tensors)

        # Handle case where backbone returns a single tensor (convert to OrderedDict)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # Convert to list as expected by head and anchor_generator
        features_list = list(features.values())

        # Get predictions from head
        head_outputs = self.head(features_list)

        # Return raw outputs without NMS post-processing
        return {
            'cls_logits': head_outputs['cls_logits'],
            'bbox_regression': head_outputs['bbox_regression'],
            'bbox_ctrness': head_outputs['bbox_ctrness']
        }

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', type=str, default='models',
                help='Path to save the exported model')
ap.add_argument('--height', type=int, default=374,
                help='Input image height')
ap.add_argument('--width', type=int, default=1238,
                help='Input image width')
args = vars(ap.parse_args())
#args = {'output': './fcos_trt_backend/models', 'height': 374, 'width': 1238}

# Create output directory if it doesn't exist
os.makedirs(args['output'], exist_ok=True)

# Load pretrained FCOS model
print("Creating pretrained FCOS backbone model...")
# Create backbone-only version
model = FCOSBackbone()
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
        'bbox_ctrness'
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
