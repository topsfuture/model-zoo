# YOLOv8模型导出
## 1. 准备工作
可选择从[YOLOV12官方主页](https://github.com/ultralytics/ultralytics?tab=readme-ov-file)下载yolov12s.pt模型，或在[导出onnx模型](#2-导出onnx模型)中自动下载模型。
安装如下依赖。

```bash
pip3 install ultralytics
```


## 2. 导出onnx模型
如果使用taNNTC编译模型，则必须先将Pytorch模型导出为onnx模型。YOL0v12官方仓库提供了模型导出接口，可以直接使用它导出onnx模型：

```python
from ultralytics import YOLO

model = YOLO("yolov12s.pt")
# Export the model to ONNX format
model.export(format="onnx",  batch=1, opset=12, dynamic=True)  # creates 'yolov12s.onnx'
```

上述脚本会在原始pt模型所在目录下生成导出的`yolov12s.onnx`模型。

