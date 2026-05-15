
# retinaface

## 目录
- [retinaface](#retinaface)
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
    - [4.4 模型精度评估](#44-模型精度评估)
---



## 1. 简介
本例程可在ea65xx平台上进行retinaface检测模型的推理，支持加载浮点和量化模型，适用于WIDERVAL数据集。模型来源：
https://github.com/biubug6/Pytorch_Retinaface

## 2. 特性
### 2.1 目录结构说明
项目目录结构如下：
```
├── cpp/         # C++例程
├── python/      # 用于量化精度校验的python脚本
├── scripts/     # 存放模型编译、数据下载、自动测试等shell脚本
├── widerface_evaluate_tps/     #推理精度评估工具
└── README.md    # 本例程的中文指南
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
下载的模型目录包括：
```
models/
├── dataset.txt
├── mobilenet0.25_Final.pth
├── mobilenetV1X0.25_pretrain.tar
├── retinaface_config_fp16.json
├── retinaface_config_int8.json
├── retinaface_float16.nb
├── retinaface_int8.nb
└── retinaface.onnx
```

下载的数据目录包括：
```
datasets/
├── face  #测试图片
└── WIDERVAL # WIDERVAL图片集
```
### 3.2 模型编译
如果您不编译模型，直接使用下载的数据集和模型，可跳过本小节。

使用taNNTC工具进行模型编译转换，具体可参考[taNNTC环境搭建](../../docs/环境安装指南.md#1-tanntc环境搭建)。环境搭建好后需在taNNTC环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译成nb。
- 生成FP16 nb
在taNNTC环境中,我们可以通过convert_model命令进行模型转换的操作，请注意修改config.json中的源模型路径、源模型框架、模型预处理参数和输入大小shape等参数，如：

进入models目录后，输入
```bash
convert_model build --output_dir ../retinaface_out  --config  ./retinaface_config_fp16.json
```
执行上述命令后，会在retinaface_out目录下生成转换好的模型文件。


- 生成INT8 nb
同上，请注意修改config.json中的源模型路径、源模型框架、模型预处理参数和输入大小shape等参数，如：
```bash
convert_model build --output_dir ../retinaface_out --config ./retinaface_config_int8.json
```

## 4. 例程测试
cpp目录下提供了C++例程以供参考使用，具体情况说明如下：
| 序号 | C++例程 | 说明 |
| ---- | ---- | ---- |
| 1 | retinaface_det_soc | 使用taOpenCV前处理，taRuntime推理 |

### 4.1 环境准备
在使用EA65xx平台时，刷机后系统已经预装了相应的taRuntime、taOpenCV库，无需额外安装，可以直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。
### 4.2 编译
通常需要在x86主机上交叉编译程序，您需要在x86上使用TACO SDK搭建交叉编译环境，具体请参考[交叉编译环境搭建](../../docs/环境安装指南.md#21-交叉编译环境搭建)。本例程主要依赖taOpencv、taRuntime等库。
交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：
```bash
cd cpp
mkdir build && cd build
cmake ..
make 
```
编译完成后，会在目录下生成retinaface_det_soc。
### 4.3 推理测试
需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到EA65XX平台测试。
#### 4.3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：
```
Usage: retinaface_det_soc [params]

        -?, -h, --help, --usage (value:true)
                print help message
        -b, --batch (value:false)
                batch mode for dataset validation
        --conf_thresh (value:0.5)
                confidence threshold for filter boxes
        -i, --input (value:test.jpg)
                input image file or directory for batch mode
        -m, --model (value:retinaface_int8.nb)
                model file
        --nms_thresh (value:0.3)
                iou threshold for nms
        -o, --output (value:output.jpg)
                output image file or json file for batch mode
        --save_result (value:false)
                save detection result images in batch mode
                
```
### 4.3.2 图片测试
图片测试实例如下。

单张图片测试：
```bash
./retinaface_det_soc --input=./datasets/face/face02.jpg --model=retinaface_float16.nb --conf_thresh=0.5 --nms_thresh=0.3 --output=output.jpg
```
测试结束后，推理标注的图片为 output.jpg，同时会打印预测结果、推理时间等信息。

也支持对整个图片文件夹进行测试:
```bash
./retinaface_det_soc --input=./datasets/face/ --model=retinaface_float16.nb --conf_thresh=0.5 --nms_thresh=0.3 --output=retinaface_float16.json --batch=true
```
测试结束后，预测的结果保存在retinaface_float16.json中，同时会打印预测结果、推理时间等信息。

### 4.4 模型精度评估
```bash
./retinaface_det_soc --input=./datasets/WIDERVAL/ --model=retinaface_float16.nb --conf_thresh=0.5 --nms_thresh=0.3 --output=WIDERVAL_retinaface_float16.json --batch=true
```
WIDERVAL集图片较多，执行时间在十几分钟左右。通过该步骤，在板端可以生成用于精度验证的json文件，我们将其拷贝至pc端，在pc端进行精度验证。工具在/retinaface/widerface_evaluate_tps/下，将板端生成的json文件复制到该目录下

先解压 widerface_txt.tar.gz
然后执行：
```bash
pip install cython-bbox
python3 ./evaluation_tps.py   -j ./WIDERVAL_retinaface_float16.json
```
执行成功后，会打印精度测试结果。




