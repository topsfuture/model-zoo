# YOLO11s-seg

## 目录
- [YOLO11_seg](#yolo11_seg)
    - [1. 简介](#1-简介)
    - [2. 特性](#2-特性)
        - [2.1 目录结构说明](#21-目录结构说明)
        - [2.2 SDK特性](#22-sdk特性)
    - [3. 数据准备与模型编译](#3-数据准备与模型编译)
        - [3.1 数据准备](#31-数据准备)
        - [3.2 模型编译](#32-模型编译)
    - [4. 例程测试](#4-例程测试)
        - [4.1 环境准备](#41-环境准备)
        - [4.2 编译](#42-编译)
        - [4.3 推理测试](#43-推理测试)
            - [4.3.1 参数说明](#431-参数说明)
            - [4.3.2 图片测试](#432-图片测试)
            - [4.3.3 JSON输出模式](#433-json输出模式)
        - [4.4 板端精度评估](#44-板端精度评估)

---

## 1. 简介
本例程可在 EA65xx 平台上进行 YOLO11s-seg 实例分割模型的推理，支持加载 FP16 模型，适用于 COCO 等常见数据集。YOLO11s-seg 是 Ultralytics 推出的轻量级实例分割模型，同时输出目标检测框和逐像素分割掩码。

模型来源：https://github.com/ultralytics/ultralytics

## 2. 特性
### 2.1 目录结构说明

项目目录结构如下：

```
├── cpp/                 # C++ 例程（板端推理）
│   ├── include/         #   头文件 (yolo11_seg.hpp, timer.hpp)
│   ├── src/             #   源文件 (yolo11_seg.cpp, main.cpp)
│   └── CMakeLists.txt   #   交叉编译配置
├── docs/                # 文档（ONNX导出指南、精度报告）
├── python/              # 模型编译/量化脚本 + 板端精度评估 (coco_eval.py)
├── scripts/             # 模型/数据下载脚本
├── models/              # ONNX 模型、NB 模型、编译配置
├── datasets/            # COCO分割数据集（coco_seg_val2017子集）
└── README.md            # 本例程的中文指南
```

### 2.2 SDK 特性
- 支持 EA6530
- 支持 FP16 模型编译和推理
- 支持图片测试（单张 / 批量目录）
- 支持 C++ 推理 + JSON 结构化输出
- 支持实例分割（检测框 + 逐像素掩码）
- 支持 NB 硬件预处理节点（自动 offload /255 归一化）

## 3. 数据准备与模型编译
### 3.1 数据准备
本例程在 scripts 目录下提供了模型和数据的下载脚本 download.sh。如果您希望自行准备模型和数据集，可跳过本小节，直接参考 [3.2 模型编译](#32-模型编译) 进行模型转换。
```bash
chmod -R +x scripts
./scripts/download.sh
```
下载的模型包括：
```
models/
├── dataset.txt
├── yolo11s_seg_float16.nb
├── yolo11s_seg.onnx
└── yolo11s_seg_config_fp16.json
```

### 3.2 模型编译
如果您不编译模型，直接使用下载的模型，可跳过本小节。

源模型需要编译成 .nb 才能在 EA65xx 平台上运行。具体可参考 [ONNX导出指南](docs/YOLO11_seg_Export_Guide.md)。

使用 taNNTC 工具进行模型编译转换，具体可参考 [taNNTC环境搭建](../../docs/环境安装指南.md#1-tanntc环境搭建)。环境搭建好后需在 taNNTC 环境中使用本例程提供的配置将 ONNX 模型编译成 .nb。

**生成 FP16 nb：**
```bash
convert_model build --output_dir ../yolo11s_seg --config models/yolo11s_seg_config_fp16.json
```

## 4. 例程测试
cpp 目录下提供了 C++ 例程以供参考使用：

| 序号 | C++ 例程 | 说明 |
|------|----------|------|
| 1 | yolo11s_seg_soc | 使用 taRuntime 推理，含前后处理，支持图片/目录批量测试 |

### 4.1 环境准备
在使用 EA65xx 平台时，刷机后系统已经预装了相应的 taRuntime、taOpenCV 库，无需额外安装。通常还需要一台 x86 主机作为开发环境，用于交叉编译 C++ 程序。

### 4.2 编译
通常需要在 x86 主机上交叉编译程序，您需要在 x86 上使用 TACO SDK 搭建交叉编译环境，具体请参考 [交叉编译环境搭建](../../docs/环境安装指南.md#21-交叉编译环境搭建)。本例程主要依赖 taOpenCV、taRuntime 等库。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：
```bash
cd cpp
mkdir build && cd build
cmake ..
make
```
编译完成后，会在目录下生成 `yolo11s_seg_soc`。

### 4.3 推理测试
需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到 EA65XX 平台测试。

#### 4.3.1 参数说明
```
Usage: yolo11s_seg_soc [params]
        -i, --input (value: test.jpg)
                test image file or directory
        -o, --output (value: output.jpg)
                output image path (for directory mode, output dir)
        -m, --model (value: yolo11s_seg_float16.nb)
                model file path
        -c, --conf (value: 0.25)
                confidence threshold
        -n, --nms (value: 0.45)
                NMS IoU threshold
        --max-det (value: 300)
                max detections per image
        --json
                output results as JSON to stdout
        --mask-dir (value: ./masks)
                directory to save per-detection mask PNGs (JSON mode)
```

#### 4.3.2 图片测试
**单张图片测试：**
```bash
./yolo11s_seg_soc -m yolo11s_seg_float16.nb -i test.jpg -o result.jpg --conf 0.25
```

**目录批量测试：**
```bash
./yolo11s_seg_soc -m yolo11s_seg_float16.nb -i input_images/ -o output_images/ --conf 0.001 --nms 0.65 --max-det 300
```
当 `-i` 参数为目录时，自动批量处理目录中所有图片（支持 .jpg/.jpeg/.png/.bmp）。

#### 4.3.3 JSON 输出模式
添加 `--json` 参数可输出结构化 JSON 结果，用于精度测试：
```bash
./yolo11s_seg_soc -m yolo11s_seg_float16.nb -i input_images/ -o output_images/ \
    --conf 0.001 --nms 0.65 --max-det 300 \
    --json --mask-dir ./masks/
```

JSON 输出格式：
```json
{
  "results": [
    {
      "image": "000000000785.jpg",
      "detections": [
        {
          "category_id": 1,
          "score": 0.85,
          "bbox": [12.3, 45.6, 100.2, 200.5],
          "mask_file": "./masks/000000000785_mask_0.png",
          "mask_size": [425, 640]
        }
      ]
    }
  ]
}
```

**注意：** `category_id` 为 YOLO 类别索引 + 1（1-80），`coco_eval.py` 内部会自动转换为 COCO 标准 category_id。

### 4.4 板端精度评估
使用 `python/coco_eval.py` 可评估板端推理的 COCO 精度指标（bbox + segm AP/AR）。

**前提条件：** 在板端使用 JSON 模式运行推理，获取结果 JSON 和 mask PNG 文件：
```bash
# 板端运行
./yolo11s_seg_soc -m yolo11s_seg_float16.nb -i input_images/ -o output_images/ \
    --conf 0.001 --nms 0.65 --max-det 300 \
    --json --mask-dir ./masks/
```

**运行评估：**
```bash
pip install pycocotools opencv-python numpy

python python/coco_eval.py \
    --board-results results.json \
    --board-mask-dir masks/ \
    --gt-ann instances_val2017.json \
    --output board_precision.txt
```

参数说明：
- `--board-results`：C++ `--json` 模式输出的 JSON 结果文件
- `--board-mask-dir`：C++ `--mask-dir` 输出的 mask PNG 目录
- `--gt-ann`：COCO GT 标注文件（`instances_val2017.json`）
- `--output`：评估报告输出路径

脚本功能：
1. 加载 COCO GT 标注
2. 解析板端 JSON 结果，将 mask PNG 编码为 RLE
3. 自动将 YOLO category_id（1-80 连续）映射为 COCO category_id（非连续）
4. 使用 pycocotools COCOeval 计算 bbox 和 segm 的 AP/AR 指标

**参考精度（400 张 COCO val2017 子集）：**

| 指标 | Board FP16 |
|------|:----------:|
| BBox AP@0.50:0.95 | 0.4837 |
| Segm AP@0.50:0.95 | 0.3453 |

完整精度对比报告见 [docs/precision_report_v2.txt](docs/precision_report_v2.txt)。
