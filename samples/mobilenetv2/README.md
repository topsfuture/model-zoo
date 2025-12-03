
# mobilenetv2

## 目录
- [mobilenetv2](#mobilenetv2)
  - [目录](#目录)
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
    - [4.3 模型精度评估](#43-模型精度评估)

---




## 1. 简介
本例程可在EA65xx平台上进行moblinetv2分类模型的推理，支持加载浮点和量化模型，适用于imagenet等常见数据集。模型来源：
https://github.com/onnx/models/blob/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/mobilenet/model/mobilenetv2-12.onnx

## 2. 特性
### 2.1 目录结构说明

项目目录结构如下：

```
├── cpp/         # C++例程
├── docs/        # 存放本例程专用文档，如onnx导出、移植常见问题等 
├── python/      # 用于量化精度校验的python脚本
├── scripts/     # 存放模型编译、数据下载、自动测试等shell脚本
└── README.md    # 本例程的中文指南
```

## 2.2 SDK特性
- 支持EA6530
- 支持FP16、INT8以及UINT8模型编译和推理
- 支持图片测试
- 支持C++推理

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
├── mobilenetv2_config_fp16.json
├── mobilenetv2_config_int8.json
├── mobilenetv2_float16.nb
├── mobilnetv2_int8.nb
└── mobilnetv2-12.onnx
```
下载的数据包括：
```
datasets/imagenet_val_1k
├── img  # imagenet中随机抽取的1000张图片
└── label.txt # imagenet中随机抽取的1000张样本对应的标注信息
```
### 3.2 模型编译
如果您不编译模型，直接使用下载的数据集和模型，可跳过本小节。
源模型需要编译成nb才能在EA65xx平台上运行，可以使用onnx模型进行编译转换,具体操作如下所述。同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。


使用taNNTC工具进行模型编译转换，具体可参考[taNNTC环境搭建](../../docs/环境安装指南.md#1-tanntc环境搭建)。环境搭建好后需在taNNTC环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译成nb。
- 生成FP16 nb
在taNNTC环境中,我们可以通过convert_model命令进行模型转换的操作，请注意修改config.json中的源模型路径、源模型框架、模型预处理参数和输入大小shape等参数，如：
```bash
convert_model build --output_dir ../mobilenet --config models/mobilenetv2_config_fp16.json
```
执行上述命令后，会在mobilenet目录下生成转换好的模型文件。
- 生成INT8 nb
同上，请注意修改config.json中的源模型路径、源模型框架、模型预处理参数和输入大小shape等参数，如：
```bash
convert_model build --output_dir ../mobilenet --config models/mobilenetv2_config_int8.json
```

## 4. 例程测试
cpp目录下提供了C++例程以供参考使用，具体情况说明如下：
| 序号 | C++例程 | 说明 |
| ---- | ---- | ---- |
| 1 | mobilenetv2_det_soc | 使用taOpencv前处理，taRuntime推理 |

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
编译完成后，会在目录下生成mobilenetv2_det_soc。
### 4.3 推理测试
需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到EA65XX平台测试。
#### 4.3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：
```
Usage: mobilenetv2_det_soc [params] 

        -?, -h, --help, --usage (value:true)
                print help message
        -a, --accuracy (value:datasets/label.txt)
                accuracy
        -i, --input_dir (value:datasets/img)
                test image dir
        -m, --model (value:mobilenetv2_int8.nb)
                model file
```
### 4.3.2 图片测试
图片测试实例如下，支持对整个图片文件夹和单张图片进行测试。
```bash
./mobilenetv2_det_soc -m=mobilenetv2_float16.nb -i=datasets/imagenet_val_1k/img/ILSVRC2012_val_00011271.JPEG
```
单张图片测试结束后，会打印预测的类别结果和概率、推理时间等信息。
```bash
./mobilenetv2_det_soc -a=datasets/imagenet_val_1k/label.txt -i=datasets/imagenet_val_1k/img/ -m=mobilenetv2_float16.nb
```
图片文件夹测试结束后，会将预测类别结果输出到result.txt中，最后在命令行中输出对应精度结果和推理时间等信息。

### 4.3 模型精度评估

模型精度评估可通过以下测试指令得到
```bash
./mobilenetv2_det_soc -a=datasets/imagenet_val_1k/label.txt -i=datasets/imagenet_val_1k/img/ -m=mobilenetv2_float16.nb
```
该指令会根据提供的数据集中的label.txt文件和推理的结果来计算模型的精度，执行成功后会在命令行中打印精度测试结果。用户如需测试个人数据集可以对应修改-a参数和-i参数。



