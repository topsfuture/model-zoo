# YOLOv8 模型推理与量化项目

## 功能特性

- **模型推理**：支持对图像进行目标检测，输出检测框、类别和置信度。
- **模型量化**：支持将 YOLOv8 模型量化为多种数据类型（如 `int8`、`float16` 等），优化模型性能。
- **批量处理**：支持批量加载图像进行推理，提高处理效率。
- **可视化结果**：将检测结果可视化并保存为图像文件。
- **灵活配置**：可通过命令行参数灵活配置模型路径、输入数据、量化类型等。

## 安装依赖

在开始之前，请确保已安装以下依赖项：

```bash
pip install numpy opencv-python tensorflow
```

## 代码结构

- **YOLOv8.py**：YOLOv8 模型类定义，包含模型初始化、推理、量化等功能。
- **infer.py**：模型推理脚本，用于对图像进行目标检测。
- **quantize.py**：模型量化脚本，用于将模型量化为指定数据类型。
- **postprocess_numpy.py**：后处理模块，包含非极大值抑制（NMS）等操作。
- **utils.py**：工具函数，如数据预处理、图像解码等。
- **results/**：推理结果保存目录，包含检测后的图像和 JSON 结果文件。
- **val2017_1000/**：示例数据集目录，用于量化和推理测试。

## 使用方法

### 模型量化

1. **准备数据集**：将用于量化的校准数据集放在指定目录（如 `val2017_1000`）。
2. **运行量化脚本**：

   ```bash
   python quantize.py --onnx_path ../yolov8s.onnx --dataset_path ../val2017_1000 --quantize_type float16 --quantize_batch_size 10
   ```

   参数说明：
   - `--onnx_path`：YOLOv8 模型的 ONNX 文件路径。
   - `--dataset_path`：校准数据集路径。
   - `--quantize_type`：量化数据类型，可选 `int8`、`float16` 等。
   - `--quantize_batch_size`：量化时每次处理的图像数量。

3. **查看结果**：量化文件将保存在当前目录下，如 `yolov8s_float16.quantize`。

### 模型推理

1. **准备输入数据**：将待检测的图像文件放在指定目录，或指定单个文件路径。
2. **运行推理脚本**：

   ```bash
   python infer.py --onnx_path ../yolov8s.onnx --dataset_path ../val2017_1000 --quantize_type float16 --batch_size 50 --output_dir ./output --output_img_dir ./output/images
   ```

   参数说明：
   - `--onnx_path`：YOLOv8 模型的 ONNX 文件路径。
   - `--dataset_path`：输入图像的路径。
   - `--quantize_type`：量化数据类型，与量化时一致。
   - `--batch_size`：推理时每次处理的图像数量。
   - `--output_dir`：推理结果保存的目录。
   - `--output_img_dir`：检测后的图像保存的目录。

3. **查看结果**：推理结果将保存在 `./output/images` 目录下，图像结果保存为图片文件，同时检测结果的详细信息将保存为 JSON 文件。

## 参数说明

- `--onnx_path`：YOLOv8 模型的 ONNX 文件路径。
- `--dataset_path`：校准数据集路径（量化时使用）或输入图像路径（推理时使用）。
- `--quantize_type`：量化数据类型，可选 `int8`、`float16` 等。
- `--batch_size`：每次处理的图像数量。
- `--output_dir`：推理结果保存的目录。
- `--output_img_dir`：检测后的图像保存的目录。
- `--quantize_batch_size`：量化时每次处理的图像数量。
- `--hybrid`：是否使用混合量化。

## 示例结果

推理完成后，检测结果将保存在 `./output` 目录下。例如：

- **图像结果**：检测后的图像将保存在 `./output/images` 目录下。
- **JSON 结果**：检测结果的详细信息将保存为 JSON 文件，文件名格式为 `yolov8s-det_opencv_python_result.json`。

## 注意事项

- 量化和推理时的 `input_shape` 和 `quantize_type` 参数应与模型实际一致。
- 数据集和输入图像的格式应为常见的图像格式（如 `.jpg`、`.png` 等）。
- 项目中使用的 YOLOv8 模型和数据集仅供参考，可根据实际需求替换为其他模型和数据。
