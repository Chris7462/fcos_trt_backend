# FCOS Object Detection TensorRT backend

The is the library for model inference using the TensorRT engine with stream.

## Generate the ONNX file
This will generate the FP32 ONNX file in the `onnxs` directory.
```bash
python3 script/export_fcos_to_onnx.py \
        --height 374 \
        --width 1238 \
        --output-dir onnxs
```

## Convert ONNX to FP16
Convert the FP32 ONNX to FP16 using ModelOpt AutoCast. Input/output tensors are kept
in FP32 for compatibility with the C++ preprocessing pipeline (`keep_io_types=True`).
```bash
python3 script/convert_onnx_to_fp16.py \
        --input  onnxs/fcos_resnet50_fpn_374x1238.onnx \
        --output onnxs/fcos_resnet50_fpn_374x1238_fp16.onnx
```

## Compile to TensorRT engine
Use trtexec to compile the FP16 ONNX to a TensorRT engine. The model is already in
FP16 at the ONNX level, so `--fp16` is not passed here.
```bash
trtexec --onnx=onnxs/fcos_resnet50_fpn_374x1238_fp16.onnx \
        --saveEngine=engines/fcos_resnet50_fpn_374x1238.engine \
        --memPoolSize=workspace:4096 \
        --verbose
```
