# YOLOv5模型导出
## 1. 准备工作
可选择从[YOLOv5官方主页](https://github.com/ultralytics/yolov5)下载yolov5s.pt模型，或在[导出onnx模型](#2-导出onnx模型)中自动下载模型。
安装如下依赖。

```bash
pip3 install ultralytics
```


## 2. 导出onnx模型
如果使用taNNTC编译模型，则必须先将Pytorch模型导出为onnx模型。可以使用如下代码将yolov5s.pt模型转换为onnx格式模型：

```python
from ultralytics import YOLO

model = YOLO("yolov5s.pt")
# Export the model to ONNX format
model.export(format="onnx", opset=17, dynamic=True)  # creates 'yolov5s.onnx'
```

上述脚本会在原始pt模型所在目录下生成导出的`yolov5s.onnx`模型。

