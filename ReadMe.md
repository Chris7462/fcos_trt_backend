```python
python3 export_to_onnx.py --output onnxs
```
This will generate the .onnx file in the onnxs folder.

Then use trtexec to compile the .onnx format to TensorRT engine
```bash
trtexec --onnx=onnxs/fcos_resnet50_fpn_374x1238.onnx --saveEngine=engines/fcos_resnet50_fpn_374x1238.engine  --memPoolSize=workspace:4096 --fp16 --verbose
```
