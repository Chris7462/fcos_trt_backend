```python
python3 export_to_onnx.py --output models
```
This will generate the .onnx file in the models folder.

Then use trtexec to compile the .onnx format to TensorRT engine
```bash
trtexec --onnx=models/fcos_resnet50_fpn_374x1238.onnx --saveEngine=engines/fcos_resnet50_fpn_374x1238.engine  --memPoolSize=workspace:4096 --fp16 --verbose
```
