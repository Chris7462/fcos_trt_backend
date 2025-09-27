#!/usr/bin/env python3
"""Script to export a pre-trained FCOS model to ONNX format for TensorRT or C++ inference."""

import argparse
from collections import OrderedDict
import os

import torch
from torch.nn import Module
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights


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
            features = OrderedDict([('0', features)])

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


def export_fcos_model(output_path, input_height, input_width):
    print('Creating pretrained FCOS backbone model...')

    # Create backbone-only version
    model = FCOSBackbone()

    print('Preparing dummy input...')
    dummy_input = ([torch.randn(3, input_height, input_width)],)

    print('Exporting to ONNX...')
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            input_names=['input'],
            output_names=[
                'cls_logits',
                'bbox_regression',
                'bbox_ctrness',
                'anchors',
                'image_sizes',
                'num_anchors_per_level'],
            verbose=True
            # Note: Dynamic axes might be tricky with list inputs and anchor generation
            # Consider using fixed batch size for more reliable ONNX export
        )
        print(f'ONNX model saved to: {output_path}')
    except Exception as e:
        print(f'✗ ONNX export failed: {e}')
        raise

    # Test the exported model
    print('\nTesting ONNX model...')
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print('✓ ONNX model validation passed')
    except ImportError:
        print('⚠ ONNX package not available - skipping model validation')
        print('  Install with: pip install onnx')
    except Exception as e:
        print(f'✗ ONNX model validation failed: {e}')


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--height', type=int, default=374, help='The height of the input image')
    ap.add_argument('--width', type=int, default=1238, help='The width of the input image')
    ap.add_argument('--output-dir', type=str, default='onnxs',
                    help='Path to save the exported model')
    args = vars(ap.parse_args())
    # args = {'height': 374, 'width': 1238, 'output_dir': 'fcos_trt_backend/onnxs'}

    # Create output directory if it doesn't exist
    os.makedirs(args['output_dir'], exist_ok=True)

    height = args['height']
    width = args['width']
    output_dir = args['output_dir']

    # Export to ONNX (backbone + heads only, no NMS)
    print(f'=== Exporting FCOS backbone for input size: {height}x{width} ===')
    output_path = os.path.join(output_dir, f'fcos_resnet50_fpn_{height}x{width}.onnx')
    export_fcos_model(output_path=output_path, input_width=width, input_height=height)

    print('ONNX export completed.')
