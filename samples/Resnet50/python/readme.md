
# Resnet50

## 项目简介

本项目实现了Resnet模型的量化和推理功能。通过使用TensorFlow和OpenCV库，您可以对图像进行预处理、量化模型并进行推理。

## 目录结构

```
.
├── Resnet.py          # Resnet模型的主要实现文件
├── quantize.py        # 量化模型的脚本
├── infer.py           # 推理脚本
├── requirements.txt   # 项目依赖
└── README.md          # 项目说明文件
```


## 使用说明

### 1. 量化模型

首先，您需要运行`quantize.py`脚本来量化模型。请确保您已经准备好要量化的ONNX模型文件。

```bash
python quantize.py --onnx_path path/to/your/model.onnx --quantize_type int8 --cali_batch_size 10
```

参数说明：
- `--onnx_path`：ONNX模型文件的路径
- `--quantize_type`：量化类型（支持`int8`、`uint8`、`int16`）
- `--batch_size`：校准批次大小

### 2. 推理

量化完成后，您可以使用`infer.py`脚本进行推理。

```bash
python infer.py --image_path path/to/your/image.jpg --model_path path/to/your/quantized_model.quantize
```

参数说明：
- `--image_path`：待推理图像的路径
- `--model_path`：量化模型文件的路径


## FAQ
混合量化：在各stage下采样和残差激活输入处推荐使用i16混合量化以提升精度。

