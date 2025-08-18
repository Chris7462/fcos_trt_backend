#!/usr/bin/env python3
"""Test script to verify ONNX FCOS model produces same results as original PyTorch model."""

import argparse
from collections import OrderedDict
import os

import onnxruntime as ort
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights


class FCOSBackbone(torch.nn.Module):
    """Wrapper to extract only the Backbone from FCOS model (same as in export script)."""

    def __init__(self):
        super().__init__()
        # Load pretrained FCOS model
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


def load_and_preprocess_image(image_path, target_height, target_width):
    """Load and preprocess image to match model input requirements."""
    print(f'Loading image: {image_path}')

    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    print(f'Original image size: {original_size}')

    # Resize to target dimensions
    image = image.resize((target_width, target_height))
    print(f'Resized to: ({target_height}, {target_width})')

    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, original_size


def print_tensor_stats(tensor, name, print_first_n=20):
    """Print statistics and first N values of a tensor."""
    flat_tensor = tensor.flatten()

    print(f'{name} (first {print_first_n} values):')
    first_values = flat_tensor[:print_first_n].detach().cpu().numpy()

    # Handle integer vs float tensors for formatting
    if tensor.dtype in [torch.int32, torch.int64, torch.long]:
        values_str = ' '.join([f'{val}' for val in first_values])
    else:
        values_str = ' '.join([f'{val:.5f}' for val in first_values])

    print(values_str)
    print(f'... (total size: {flat_tensor.numel()})')

    print(f'{name} Statistics:')
    if tensor.dtype in [torch.int32, torch.int64, torch.long]:
        print(f'  Min: {flat_tensor.min()}')
        print(f'  Max: {flat_tensor.max()}')
        print(f'  Mean: {flat_tensor.float().mean():.5f}')
    else:
        print(f'  Min: {flat_tensor.min():.5f}')
        print(f'  Max: {flat_tensor.max():.5f}')
        print(f'  Mean: {flat_tensor.mean():.5f}')
    print(f'  Size: {flat_tensor.numel()}')


def run_pytorch_inference(model, image_tensor):
    """Run inference using PyTorch model."""
    print('\n=== RUNNING PYTORCH INFERENCE ===')

    with torch.no_grad():
        outputs = model(image_tensor)

    print('=== PYTORCH INFERENCE RESULTS ===')
    print_tensor_stats(outputs['cls_logits'], 'CLS_LOGITS')
    print_tensor_stats(outputs['bbox_regression'], 'BBOX_REGRESSION')
    print_tensor_stats(outputs['bbox_ctrness'], 'BBOX_CTRNESS')
    print_tensor_stats(outputs['anchors'], 'ANCHORS')
    print_tensor_stats(outputs['image_sizes'], 'IMAGE_SIZES')
    print_tensor_stats(outputs['num_anchors_per_level'], 'NUM_ANCHORS_PER_LEVEL')
    print('✓ PyTorch inference completed successfully!')

    return outputs


def run_onnx_inference(onnx_path, image_tensor):
    """Run inference using ONNX model."""
    print('\n=== RUNNING ONNX INFERENCE ===')
    print(f'Loading ONNX model from: {onnx_path}')

    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)

    # Get input name
    input_name = ort_session.get_inputs()[0].name

    # Convert tensor to numpy
    image_np = image_tensor.detach().cpu().numpy()

    # Run inference
    ort_inputs = {input_name: image_np}
    ort_outputs = ort_session.run(None, ort_inputs)

    # Convert outputs back to tensors for comparison
    outputs = {
        'cls_logits': torch.from_numpy(ort_outputs[0]),
        'bbox_regression': torch.from_numpy(ort_outputs[1]),
        'bbox_ctrness': torch.from_numpy(ort_outputs[2]),
        'anchors': torch.from_numpy(ort_outputs[3]),
        'image_sizes': torch.from_numpy(ort_outputs[4]),
        'num_anchors_per_level': torch.from_numpy(ort_outputs[5])
    }

    print('=== ONNX INFERENCE RESULTS ===')
    print_tensor_stats(outputs['cls_logits'], 'CLS_LOGITS')
    print_tensor_stats(outputs['bbox_regression'], 'BBOX_REGRESSION')
    print_tensor_stats(outputs['bbox_ctrness'], 'BBOX_CTRNESS')
    print_tensor_stats(outputs['anchors'], 'ANCHORS')
    print_tensor_stats(outputs['image_sizes'], 'IMAGE_SIZES')
    print_tensor_stats(outputs['num_anchors_per_level'], 'NUM_ANCHORS_PER_LEVEL')
    print('✓ ONNX inference completed successfully!')

    return outputs


def compare_outputs(pytorch_outputs, onnx_outputs, tolerance=1e-4):
    """Compare PyTorch and ONNX outputs."""
    print(f'\n=== COMPARING OUTPUTS (tolerance: {tolerance}) ===')

    all_close = True

    for key in pytorch_outputs.keys():
        pytorch_tensor = pytorch_outputs[key]
        onnx_tensor = onnx_outputs[key]

        # Check if shapes match
        if pytorch_tensor.shape != onnx_tensor.shape:
            print(f'✗ {key}: Shape mismatch! PyTorch: {pytorch_tensor.shape},'
                  f'ONNX: {onnx_tensor.shape}')
            all_close = False
            continue

        # Check if values are close
        is_close = torch.allclose(pytorch_tensor, onnx_tensor, atol=tolerance, rtol=tolerance)

        if is_close:
            print(f'✓ {key}: Values match within tolerance')
        else:
            print(f'✗ {key}: Values do NOT match!')

            # Print difference statistics
            diff = torch.abs(pytorch_tensor - onnx_tensor)
            print(f'  Max absolute difference: {diff.max():.8f}')
            print(f'  Mean absolute difference: {diff.mean():.8f}')

            # Print some example differences
            flat_pytorch = pytorch_tensor.flatten()
            flat_onnx = onnx_tensor.flatten()
            flat_diff = diff.flatten()

            # Find indices of largest differences
            _, max_diff_indices = torch.topk(flat_diff, min(5, flat_diff.numel()))

            print('  Largest differences:')
            for i, idx in enumerate(max_diff_indices[:5]):
                idx = idx.item()
                print(f'    [{idx}]: PyTorch={flat_pytorch[idx]:.8f}, ONNX={flat_onnx[idx]:.8f},'
                      f'diff={flat_diff[idx]:.8f}')

            all_close = False

    if all_close:
        print('\n🎉 SUCCESS: All outputs match between PyTorch and ONNX models!')
    else:
        print('\n❌ FAILURE: Outputs differ between PyTorch and ONNX models!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ONNX FCOS model against PyTorch original')
    parser.add_argument('--onnx-path', type=str, required=True,
                        help='Path to the ONNX model file')
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to test image')
    parser.add_argument('--height', type=int, default=374,
                        help='Input height (must match ONNX model)')
    parser.add_argument('--width', type=int, default=1238,
                        help='Input width (must match ONNX model)')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Tolerance for output comparison')

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.onnx_path):
        print(f'Error: ONNX model not found at {args.onnx_path}')
        raise FileNotFoundError(f'ONNX model not found at {args.onnx_path}')

    if not os.path.exists(args.image_path):
        print(f'Error: Test image not found at {args.image_path}')
        raise FileNotFoundError(f'Test image not found at {args.image_path}')

    print(f'Loaded image: {args.image_path} (height, width) = ({args.height}, {args.width})')

    # Load and preprocess image
    image_tensor, original_size = load_and_preprocess_image(
        args.image_path, args.height, args.width
    )

    print('Running inference...')

    # Create PyTorch model
    print('\nCreating PyTorch FCOS model...')
    pytorch_model = FCOSBackbone()

    # Run PyTorch inference
    pytorch_outputs = run_pytorch_inference(pytorch_model, image_tensor)

    # Run ONNX inference
    onnx_outputs = run_onnx_inference(args.onnx_path, image_tensor)

    # Compare outputs
    compare_outputs(pytorch_outputs, onnx_outputs, args.tolerance)
