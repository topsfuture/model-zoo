# Insightface

## 目录
- [Insightface](#insightface)
    - [1. 简介](#1-简介)
    - [2. 特性](#2-特性)
        - [2.1 目录结构说明](#21-目录结构说明)
        - [2.2 特性](#22-特性)
    - [3. 数据准备与模型编译](#3-数据准备与模型编译)
        - [3.1 数据准备](#31-数据准备)
        - [3.2 模型编译](#32-模型编译)
    - [4. 例程测试](#4-例程测试)
        - [4.1 环境准备](#41-环境准备)
        - [4.2 编译](#42-编译)
        - [4.3 推理测试](#43-推理测试)
            - [4.3.1 参数说明](#431-参数说明)
            - [4.3.2 运行测试](#432-运行测试)



---

## 1. 简介
本例程基于 Insightface 模型，经过优化和适配，可在EA65xx硬件平台上高效运行，Insightface 是一个领先的人脸识别和分析开源工具包，包含人脸检测、人脸对齐和人脸识别等功能。本例程实现了基于 SCRFD 的人脸检测和基于 ArcFace 的人脸识别，支持在嵌入式平台上实时运行。


## 2. 特性

### 2.1 目录结构说明

```
├── cpp/        # C++ 例程
├── python/     # python 例程
├── docs/       # 存放本例程专用文档，如onnx导出、移植常见问题等 
├── scripts/    # 存放模型编译、数据下载、自动测试等shell脚本
└── README.md   # 本例程的中文指南

```

### 2.2 特性

- 支持在 EA65xx 硬件平台上运行
- 使用taRuntime进行模型推理
- 提供 float16、int8 量化模型(`.nb`格式)
- 包含使用taOpenCV进行预处理和后处理的C++示例代码
- 支持 LFW 数据集的人脸识别精度评估
- 应用场景：人脸门禁、人脸考勤、智能监控、人脸搜索
- 技术领域：人脸检测、人脸识别、计算机视觉


## 3. 数据准备与模型编译
### 3.1 数据准备
本例程在 scripts 目录下提供了模型和数据的下载脚本 download.sh。如果您希望自行准备模型和数据集，可跳过本小节，直接参考 [3.2 模型编译](#32-模型编译) 进行模型转换。
```bash
chmod +x scripts/download.sh
./scripts/download.sh
```

下载内容：
```bash
.
├── datasets
│   ├── LFW  # LFW数据集
│   └── lfw_pairs.txt
├── models
│   ├── dataset.txt  # det 网络 scrfd 的量化数据
│   ├── det_500m_config_float16.json
│   ├── det_500m_config_int8.json
│   ├── det_500m_float16.nb
│   ├── det_500m_int8.nb
│   ├── det_500m.onnx  # 人脸检测模型（SCRFD-500M)
│   ├── mbf_quant_all.npy   # mbf 网络的量化数据
│   ├── w600k_mbf_config_float16.json
│   ├── w600k_mbf_config_int8.json
│   ├── w600k_mbf_float16.nb
│   ├── w600k_mbf_int8.nb
│   └── w600k_mbf.onnx  # 人脸识别模型（MobileFaceNet）
└── test_image
    └── test_image.jpg
```

其中 mbf 网络的量化数据，通过执行 Model_Zoo 代码仓库中的 python/mbf_npy_generate.py 获得：
```bash
cd python/
python3 mbf_npy_generate.py --data-dir ../datasets/LFW --output-dir ./mbf_quant_data --num-samples 10
```



### 3.2 模型编译
如果您不编译模型，直接使用下载的数据集和模型，可跳过本小节。
源模型需要编译成nb才能在EA65xx平台上运行，可以使用onnx模型进行编译转换。具体可参考[模型转换](docs/Insightface_Export_Guide.md)。同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。


使用taNNTC工具进行模型编译转换，具体可参考[taNNTC环境搭建](../../docs/环境安装指南.md#1-tanntc环境搭建)。环境搭建好后需在taNNTC环境中进入例程的 models/ 目录，并使用本例程提供的命令将onnx模型编译成nb。

- 生成FP16 nb
在taNNTC环境中,我们可以通过convert_model命令进行模型转换的操作，请注意修改config.json中的源模型路径、源模型框架、模型预处理参数和输入大小shape等参数，如：

```bash
# 转换 det 模型
convert_model build --output_dir ./output/ --config det_500m_config_float16.json
# 转换 rec 模型
convert_model build --output_dir ./output/ --config w600k_mbf_config_float16.json
```
执行上述命令，会在 ./output/ 目录下生成转换好的 .nb 模型文件。
- 生成INT8 nb
同上，请注意修改config.json中的源模型路径、源模型框架、模型预处理参数和输入大小shape等参数，如：
```bash
# 转换 det 模型
convert_model build --output_dir ./output/ --config det_500m_config_int8.json
# 转换 rec 模型
convert_model build --output_dir ./output/ --config w600k_mbf_config_int8.json
```


## 4. 例程测试
cpp目录下提供了C++例程以供参考使用，具体情况说明如下：
| 序号 | C++例程 | 说明 |
| ---- | ---- | ---- |
| 1 | insightface_soc | 使用taOpenCV前处理，taRuntime推理 |

### 4.1 环境准备
在使用EA65xx平台时，刷机后系统已经预装了相应的taRuntime、taOpenCV库，无需额外安装，可以直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。
### 4.2 编译
通常需要在x86主机上交叉编译程序，您需要在x86上使用TACO SDK搭建交叉编译环境，具体请参考[交叉编译环境搭建](../../docs/环境安装指南.md#21-交叉编译环境搭建)。本例程主要依赖taOpenCV、taRuntime等库。
交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
cd cpp
mkdir build && cd build
cmake ..
make 
```
编译完成后，会在目录下生成insightface_soc。
### 4.3 推理测试
需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到EA65XX平台测试。
#### 4.3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：

```bash
Usage: ./insightface_soc [OPTIONS]
Options:
  --det-model NAME   Detection model base name (default: models/det_500m_float16.nb)
  --rec-model NAME   Recognition model base name (default: models/w600k_mbf_float16.nb)
  --image PATH      Input image path (for single image mode)
  --eval            Enable LFW evaluation mode
  --lfw-data DIR    LFW dataset directory (default: ./dataset/LFW)
  --lfw-pairs FILE  LFW pairs.txt file (default: ./dataset/lfw_pairs.txt)
  --core-id ID      NPU core ID (0 or 1)
  --score-thresh F  Score threshold (default: 0.5)
  --nms-thresh F    NMS threshold (default: 0.4)
  --profile BOOL    Enable performance profiling (default: true)
  --silent BOOL     Enable silent mode (no progress output) (default: false)
  --help            Show this help

```


### 4.3.2 运行测试
例程单张测试执行如下命令：
```bash
chmod +x insightface_soc

./insightface_soc --det-model models/det_500m_float16.nb --rec-model models/w600k_mbf_float16.nb --image ./dataset/test_images/test_image.jpg 
```
LFW 数据集精度评估执行命令：

```bash
chmod +x insightface_soc

./insightface_soc --det-model models/det_500m_float16.nb --rec-model models/w600k_mbf_float16.nb --eval
```
