import onnx
import sys

def inspect_onnx_model(onnx_path):
    """Inspect ONNX model to understand its structure"""
    print(f"Inspecting ONNX model: {onnx_path}")

    # Load the model
    model = onnx.load(onnx_path)

    print(f"\n=== Model Info ===")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    print(f"Domain: {model.domain}")

    graph = model.graph
    print(f"\n=== Graph Info ===")
    print(f"Name: {graph.name}")
    print(f"Nodes: {len(graph.node)}")

    print(f"\n=== Inputs ===")
    for i, input_tensor in enumerate(graph.input):
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"  {i}: {input_tensor.name} - {shape}")

    print(f"\n=== Outputs ===")
    expected_outputs = {'cls_logits', 'bbox_regression', 'bbox_ctrness', 'anchors',
                       'image_sizes', 'original_image_sizes', 'num_anchors_per_level'}

    valid_outputs = []
    unexpected_outputs = []

    for i, output_tensor in enumerate(graph.output):
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in output_tensor.type.tensor_type.shape.dim]

        if output_tensor.name in expected_outputs:
            valid_outputs.append((output_tensor.name, shape))
            print(f"  ✓ {i}: {output_tensor.name} - {shape}")
        else:
            unexpected_outputs.append((output_tensor.name, shape))
            print(f"  ✗ {i}: {output_tensor.name} - {shape} (UNEXPECTED)")

    print(f"\n=== Summary ===")
    print(f"Valid outputs: {len(valid_outputs)}")
    print(f"Unexpected outputs: {len(unexpected_outputs)}")

    if unexpected_outputs:
        print(f"\n=== Issue Detected ===")
        print(f"Your ONNX model has {len(unexpected_outputs)} unexpected outputs.")
        print("This is causing the TensorRT engine to have extra tensors.")
        print("You need to re-export your ONNX model to fix this.")

        print(f"\nUnexpected output details:")
        for name, shape in unexpected_outputs:
            print(f"  - {name}: {shape}")
    else:
        print(f"\n✓ ONNX model looks good - all outputs are expected!")

    return len(unexpected_outputs) == 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_onnx.py <path_to_onnx_file>")
        sys.exit(1)

    onnx_path = sys.argv[1]
    is_valid = inspect_onnx_model(onnx_path)

    if not is_valid:
        print(f"\nRecommendation: Use the fixed export script to re-export your model.")
        sys.exit(1)
    else:
        print(f"\n✓ Model is ready for TensorRT conversion!")
