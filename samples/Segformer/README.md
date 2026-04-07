# Segformer

## 目录
- [Segformer](#segformer)
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
            - [4.3.2 运行结果](#432-运行结果)
        - [4.4 模型精度评估](#44-模型精度评估)


---

## 1. 简介
本例程基于 Segformer 模型，经过优化和适配，可在EA65xx硬件平台上高效运行，Segformer 采用 Transformer 编码器与轻量级 MLP 解码器结构，能够在保持高精度的同时实现快速推理，适用于实时语义分割任务。


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
- 提供 float16 量化模型(`.nb`格式)
- 包含使用taOpenCV进行预处理和后处理的C++示例代码
- 支持常见语义分割数据集类别输出
- 应用场景：自动驾驶场景理解、遥感图像分割、实时视频分割
- 技术领域：语义分割、Transformer、轻量级网络


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
│   └── cityscapes  # cityscapes 数据集抽取500样本
└── models
    ├── segformer_config.json  # 模型转换的配置文件
    ├── segformer_float16.nb  # 转换后的float16模型
    └── segformer_sim.onnx  # 经过onnx simplify的onnx模型

```




### 3.2 模型编译
如果您不编译模型，直接使用下载的数据集和模型，可跳过本小节。
源模型需要编译成nb才能在EA65xx平台上运行，可以使用onnx模型进行编译转换。具体可参考[模型转换](docs/Segformer_Export_Guide.md)。同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。


使用taNNTC工具进行模型编译转换，具体可参考[taNNTC环境搭建](../../docs/环境安装指南.md#1-tanntc环境搭建)。环境搭建好后需在taNNTC环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译成nb。

- 生成FP16 nb
在taNNTC环境中,我们可以通过convert_model命令进行模型转换的操作，请注意修改config.json中的源模型路径、源模型框架、模型预处理参数和输入大小shape等参数，如：

```bash
convert_model build --output_dir ../segformer/ --config segformer_config.json
```
注：Segformer 模型导出时，需要在 export 字段配置`VIV_VX_ENABLE_LAYOUT_OPT`环境变量如下：
```bash
      "env" : {
        "VIV_VX_ENABLE_LAYOUT_OPT": "1"
        }
```


## 4. 例程测试
cpp目录下提供了C++例程以供参考使用，具体情况说明如下：
| 序号 | C++例程 | 说明 |
| ---- | ---- | ---- |
| 1 | segformer_soc | 使用taOpenCV前处理，taRuntime推理 |

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
编译完成后，会在目录下生成segformer_soc。
### 4.3 推理测试
需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到EA65XX平台测试。
#### 4.3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：

```bash
Usage: segformer_soc [params] 

	-?, -h, --help, --usage (value:true)
		print help message
	-b, --batch (value:true)
		run in batch mode (process all images in dataset)
	-c, --core_id (value:0)
		core ID for execution (0 or 1)
	-d, --dataset_path (value:./dataset)
		path to the image/dataset directory
	-m, --model_path (value:./models/segformer_float16.nb)
		path to the model file
	-o, --output_dir (value:./outputs)
		directory to save results

```


### 4.3.2 运行测试
例程测试如下：
```bash
chmod +x segformer_soc

./segformer_soc -m=models/segformer_float16.nb
```



## 4.4 模型精度评估
执行上一步后，将在目录下生成输出目录 outputs/，包含了分割图和 .json 文件，供精度评估工具使用。将 outputs/ 目录拷贝至 segformer/ 目录，精度评估需要数据集标注文件，根据实际调整 datasets/ 目录的路径。执行评估脚本：

```bash
python3 python/segformer_eval.py --result_json outputs/segformer_float16_outputs_opencv_cpp_result.json
```

