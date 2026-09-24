# YOLO11s-seg 模型导出与编译

## 1. 准备工作

从 [Ultralytics 官方仓库](https://github.com/ultralytics/ultralytics) 下载 `yolo11s-seg.pt` 模型，或在 [导出 ONNX 模型](#2-导出-onnx-模型) 中自动下载。

安装依赖：

```bash
pip3 install ultralytics torch
```

## 2. 导出 ONNX 模型

如果使用 taNNTC 编译模型，必须先将 PyTorch 模型导出为 ONNX 格式。本例程提供了导出脚本：

```bash
cd python
python export_onnx.py
```

或手动导出：

```python
from ultralytics import YOLO

model = YOLO("yolo11s-seg.pt")
model.export(format="onnx", opset=14, imgsz=640)
```

导出后在 `../models/` 目录下生成 `yolo11s_seg.onnx`。
