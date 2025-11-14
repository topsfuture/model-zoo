
# YOLO11

## 目录
- [YOLO11](#yolo11)
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
本例程可在ea65xx平台上进行YOLO11检测模型的推理，支持加载浮点和量化模型，适用于COCO等常见数据集。模型来源：
https://github.com/airockchip/ultralytics_yolo11

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
├── datasets.txt
├── yolo11s_float16.nb
├── yolo11s.onnx
├── yolo11s_int8.nb
├── yolo11s_config_fp16.json
└── yolo11s_config_int8.json
```
下载的数据包括：
```
datasets/
├── val2017_1000  # coco val2017中随机抽取的1000张样本
└── instances_val2017_1000.json # coco val2017中随机抽取的1000张样本对应的标注信息
```
### 3.2 模型编译
如果您不编译模型，直接使用下载的数据集和模型，可跳过本小节。
源模型需要编译成nb才能在EA65xx平台上运行，可以使用onnx模型或者torchscripts模型进行编译转换。具体可参考[模型转换](docs/YOLO11_Export_Guide.md)。同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。


使用taNNTC工具进行模型编译转换，具体可参考[taNNTC环境搭建](../../docs/环境安装指南.md#1-tanntc环境搭建)。环境搭建好后需在taNNTC环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译成nb。
- 生成FP16 nb
在taNNTC环境中,我们可以通过convert_model命令进行模型转换的操作，请注意修改config.json中的源模型路径、源模型框架、模型预处理参数和输入大小shape等参数，如：
```bash
convert_model build --output_dir ../yolo11 --config models/yolo11s_config_fp16.json
```
执行上述命令后，会在yolo11目录下生成转换好的模型文件。
- 生成INT8 nb
同上，请注意修改config.json中的源模型路径、源模型框架、模型预处理参数和输入大小shape等参数，如：
```bash
convert_model build --output_dir ../yolo11 --config models/yolo11s_config_int8.json
```

## 4. 例程测试
cpp目录下提供了C++例程以供参考使用，具体情况说明如下：
| 序号 | C++例程 | 说明 |
| ---- | ---- | ---- |
| 1 | yolo11s_det_soc | 使用ta_opencv前处理，taruntime推理 |

### 4.1 环境准备
在使用EA65xx平台时，刷机后系统已经预装了相应的taruntime、ta-opencv库，无需额外安装，可以直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。
### 4.2 编译
通常需要在x86主机上交叉编译程序，您需要在x86上使用TACO SDK搭建交叉编译环境，具体请参考[交叉编译环境搭建](../../docs/环境安装指南.md#21-交叉编译环境搭建)。本例程主要依赖ta-opencv、ta-runtime等库。
交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：
```bash
cd cpp
mkdir build && cd build
cmake ..
make 
```
编译完成后，会在目录下生成yolo11s_det_soc。
### 4.3 推理测试
需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到EA65XX平台测试。
#### 4.3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：
```
Usage: yolo11s_det_soc [params] 
        --input (value:test.jpg)
                input image file or directory for batch mode
        --model (value:yolo11s_int8.nb)
                model file path
        --output (value:output.jpg)
                output image file or json file for batch mode
        --conf_thresh (value:0.25)
                confidence threshold for filter boxes
        --nms_thresh (value:0.45)
                iou threshold for nms
        --batch (value:false)
                batch mode for dataset validation
        --save_result (value:false)
                save detection result images in batch mode
        --help (value:true)
                print help information
```
### 4.3.2 图片测试
图片测试实例如下，支持对整个图片文件夹进行测试。
```bash
./yolo11s_det_soc --input=./datasets/val2017_1000/ --model=yolo11s_float16.nb --conf_thresh=0.001 --nms_thresh=0.6 --output=yolo11s_fp16.json --batch=true
```
测试结束后，会将预测的图片保存在results/下，预测的结果保存在yolo11s_fp16.json中，同时会打印预测结果、推理时间等信息。

### 4.3 模型精度评估

通过以上步骤的，在板端可以获得用于精度验证的json文件，我们需要将其拷贝至pc端。精度验证在pc端进行，验证的脚本为YOLO11/python/utils/coco_eval.py，真实标注json文件在datasets下（instances_val2017_1000.json），具体执行路径请用户自行指定。

```bash
pip install pycocotools
python3 coco_eval.py -g 用户路径/instances_val2017_1000.json -r 用户路径/yolo11s_fp16.json
```
执行成功后，会打印精度测试结果。




