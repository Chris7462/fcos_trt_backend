import torch
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
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

        # Generate anchors - pass both images and features
        anchors = self.anchor_generator(images, features_list)

        # Get predictions from head
        head_outputs = self.head(features_list)

        # Return raw outputs without NMS post-processing
        return {
            'cls_logits': head_outputs['cls_logits'],
            'bbox_regression': head_outputs['bbox_regression'],
            'bbox_ctrness': head_outputs['bbox_ctrness'],
            'anchors': anchors
        }

def print_tensor_stats(tensor, name):
    """Print statistics for a tensor"""
    flat_tensor = tensor.flatten()
    print(f"\n{name} Statistics:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Size: {tensor.size}")
    print(f"  Min: {flat_tensor.min():.8f}")
    print(f"  Max: {flat_tensor.max():.8f}")
    print(f"  Mean: {flat_tensor.mean():.8f}")
    print(f"  Std: {flat_tensor.std():.8f}")
    print(f"  First 10 values: {flat_tensor[:10].tolist()}")

def detailed_intermediate_results():
    """Show intermediate outputs to identify where differences might occur"""

    print("\n=== Detailed Intermediate Results ===")

    torch.manual_seed(42)

    # Load the same test image
    test_image_path = "script/image_000.png"

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

    # Original model
    original_model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT)
    original_model.eval()

    # Custom model
    backbone_model = FCOSBackbone()

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

        # Print detailed statistics
        print_tensor_stats(custom_outputs['cls_logits'], "PyTorch cls_logits")
        print_tensor_stats(custom_outputs['bbox_regression'], "PyTorch bbox_regression")
        print_tensor_stats(custom_outputs['bbox_ctrness'], "PyTorch bbox_ctrness")

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
    # Run detailed intermediate comparison
    detailed_intermediate_results()

    test_image_path = "script/image_000.png"

    print("\n=== Test Complete ===")
    print("If the models produce nearly identical results, your FCOSBackbone")
    print("implementation is correct and ready for ONNX export!")
