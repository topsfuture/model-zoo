# CLIP

## 目录
- [CLIP](#clip)
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
        - [4.4 模型精度评估](#44-模型精度评估)


---

## 1. 简介
本例程基于OpenAI CLIP（Contrastive Language-Image Pre-training）模型，经过优化和适配，可在EA65xx硬件平台上高效运行，用于实现图像与文本的多模态理解与检索。CLIP模型能够将图像和文本映射到统一的特征空间，计算它们之间的相似度。

项目代码通过taRuntime加载并执行.nb模型，利用taOpenCV进行图像的预处理和后处理，支持双向检索（图像到文本、文本到图像）。

## 2. 特性
### 2.1 目录结构说明

项目目录结构如下：

```
├── cpp/        # C++例程
├── docs/       # 存放本例程专用文档，如onnx导出、移植常见问题等 
├── python/     # 存放onnx模型的python推理脚本
├── scripts/    # 存放模型编译、数据下载、自动测试等shell脚本
└── README.md   # 本例程的中文指南

```

## 2.2 SDK特性
- 支持EA6530
- 支持FP16、INT8模型编译和推理
- 支持图片测试
- 支持C++推理

## 3. 数据准备与模型编译
### 3.1 数据准备
本例程在 scripts 目录下提供了模型和数据的下载脚本 download.sh。如果您希望自行准备模型和数据集，可跳过本小节，直接参考 [3.2 模型编译](#32-模型编译) 进行模型转换。
```bash
chmod -R +x scripts
./scripts/download.sh
```
下载的模型包括:
```
models/
├── clip_image_config.json
├── clip_image_float16.nb
├── clip_image_vitb32.onnx
├── clip_text_config.json
├── clip_text_float16.nb
└── clip_text_vitb32.onnx
```
下载的精度测试数据包括:
```
datasets/
└── test.bin  # CIFAR_100中10000张图片以及对应的标注信息
```
下载的测试图片数据包括:
```
test_image/
├── Car-headlights-misidentified-as-flames.jpg
├── CLIP.png
└── Clothes-and-hats-misidentified-as-safety-helmet.jpg
```
### 3.2 模型编译
本例程需要准备两个模型文件: image_encoder 模型和 text_encoder 模型。
如果您不编译模型，直接使用下载的数据集和模型，可跳过本小节。
源模型需要编译成nb才能在EA65xx平台上运行，可以使用onnx模型或者torchscripts模型进行编译转换。具体可参考[模型转换](docs/CLIP_Export_Guide.md)。同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。


使用taNNTC工具进行模型编译转换，具体可参考[taNNTC环境搭建](../../docs/环境安装指南.md#1-tanntc环境搭建)。环境搭建好后需在taNNTC环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译成nb。

- 生成FP16 nb
在taNNTC环境中,我们可以通过convert_model命令进行模型转换的操作，请注意修改config.json中的源模型路径、源模型框架、模型预处理参数和输入大小shape等参数，如：

```bash
convert_model build --output_dir ../clip/ --config clip_image_config.json
```
text_encoder 网络也是同样转换：
```bash
convert_model build --output_dir ../clip/ --config clip_text_config.json
```


## 4. 例程测试
cpp目录下提供了C++例程以供参考使用，具体情况说明如下：
| 序号 | C++例程 | 说明 |
| ---- | ---- | ---- |
| 1 | clip_soc | 使用taOpenCV前处理，taRuntime推理 |

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
编译完成后，会在目录下生成clip_soc。
### 4.3 推理测试
需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到EA65XX平台测试。
#### 4.3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：

```bash
Usage: clip_soc [params] 

        -?, -h, --help, --usage (value:true)
                print help message
        --image_model (value:./models/clip_image_float16.nb)
                path to the image model file
        --image_path (value:./dataset)
                path to the image directory
        --text (value:"a diagram,a person ,a Car")
                text inputs for prediction (multiple texts can be separated by spaces and must be quoted)
        --text_model (value:./models/clip_text_float16.nb)
                path to the text model file
        --text_projection_path (value:./text_projection_512_512.npy)
                path to the text projection file
```


### 4.3.2 图片测试
例程测试如下：
```bash
./clip_soc --image_model=models/clip_image_float16.nb --image_path=test_image/ --text_model=models/clip_text_float16.nb --text="a diagram,a person,a Car" --text_projection_path=text_projection_512_512.npy
```
测试结束后，会打印图片路径中的每一张图片与输入文本的相似度以及文本与图片的相似度

## 4.4 模型精度评估

修改 CMakeLists.txt，编译 cifar100_validator 精度校验程序
```bash
set(SRC_clip
    clip.cpp
    cifar100_validator.cpp)
set(EXE_clip cifar100_validator)
add_executable(${EXE_clip} ${SRC_clip})
target_link_libraries(${EXE_clip} ${OPENCV_LIBS} ${OTHER_LIBS}  ${VIPLITE_LIBS} -ldl -fprofile-arcs -lgcov -lpthread -lta_runtime)
```

在 EA65xx 平台上推理 CIFAR-100 数据集，可通过以下测试指令进行:
```bash
./cifar100_validator ./models/clip_image_float16.nb ./models/clip_text_float16.nb ./text_projection_512_512.npy ./datasets/test.bin 1000
```
该指令会解析test.bin中提供的图片以及真实标签来得到模型精度，其中参数1000指代测试1000张图片，执行成功后会在命令行中打印Top-1和Top-5精度结果,并将具体结果保存在cifar100_fine_validation_results.txt文件中。
