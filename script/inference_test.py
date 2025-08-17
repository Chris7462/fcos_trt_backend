import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.context.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, input_data):
        # Copy input data to host buffer
        np.copyto(self.inputs[0]['host'], input_data.ravel())

        # Create CUDA stream
        stream = cuda.Stream()

        # Transfer input data to device
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], stream)

        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=stream.handle)

        # Transfer outputs back to host
        outputs = []
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], stream)
            stream.synchronize()
            outputs.append(output['host'].copy())

        return outputs

def compare_models():
    """Compare PyTorch model with TensorRT engine"""

    # Load original PyTorch model
    pytorch_model = torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT")
    pytorch_model.eval()

    # Load TensorRT engine
    trt_inference = TRTInference("engines/fcos_trt_nms.engine")

    # Create test input
    test_input = torch.randn(1, 3, 374, 1238)

    # PyTorch inference
    with torch.no_grad():
        pytorch_outputs = pytorch_model(test_input)
        pytorch_boxes = pytorch_outputs[0]['boxes'].numpy()
        pytorch_labels = pytorch_outputs[0]['labels'].numpy()
        pytorch_scores = pytorch_outputs[0]['scores'].numpy()

    # TensorRT inference
    trt_outputs = trt_inference.infer(test_input.numpy())
    trt_boxes = trt_outputs[0].reshape(-1, 4)  # Adjust shape as needed
    trt_labels = trt_outputs[1].reshape(-1)    # Adjust shape as needed
    trt_scores = trt_outputs[2].reshape(-1)    # Adjust shape as needed

    # Filter out empty detections
    valid_mask = trt_scores > 0.1
    trt_boxes = trt_boxes[valid_mask]
    trt_labels = trt_labels[valid_mask]
    trt_scores = trt_scores[valid_mask]

    print(f"PyTorch detections: {len(pytorch_boxes)}")
    print(f"TensorRT detections: {len(trt_boxes)}")

    print(f"\nPyTorch top 5 scores: {pytorch_scores[:5]}")
    print(f"TensorRT top 5 scores: {trt_scores[:5]}")

    # Check if results are similar (allowing for some numerical differences)
    if len(pytorch_boxes) > 0 and len(trt_boxes) > 0:
        score_diff = np.abs(pytorch_scores[:min(5, len(pytorch_scores))] -
                           trt_scores[:min(5, len(trt_scores))])
        print(f"Score differences (top 5): {score_diff}")

        if np.all(score_diff < 0.01):  # Threshold for acceptable difference
            print("✅ Models produce similar results!")
        else:
            print("⚠️ Significant differences detected - check implementation")

def benchmark_inference():
    """Benchmark TensorRT engine performance"""
    import time

    trt_inference = TRTInference("engines/fcos_trt_nms.engine")
    test_input = torch.randn(1, 3, 374, 1238)

    # Warmup
    for _ in range(10):
        _ = trt_inference.infer(test_input.numpy())

    # Benchmark
    num_runs = 100
    start_time = time.time()

    for _ in range(num_runs):
        _ = trt_inference.infer(test_input.numpy())

    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_time

    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    print("Testing TensorRT engine...")
    compare_models()

    print("\nBenchmarking performance...")
    benchmark_inference()
