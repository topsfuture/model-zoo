
# FastVLM

## 目录
- [FastVLM](#FastVLM)
  - [目录](#目录)
  - [1. 模型简介](#1-模型简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
  - [3. 模型准备](#3-模型准备)
    - [3.1 使用提供的模型](#31-使用提供的模型)
    - [3.2 自行编译模型](#32-自行编译模型)
      - [3.2.1 下载官方仓库源模型](#321-下载官方仓库源模型)
      - [3.2.2 准备taNNTC工具](#322-准备tanntc工具)
      - [3.2.3 准备量化图片](#323-准备量化图片)
      - [3.2.4 执行编译过程](#324-执行编译过程)
      - [3.2.5 对权重文件进行裁剪](#325-对权重文件进行裁剪)
  - [4. 模型使用](#4-模型使用)
    - [4.1 环境准备](#41-环境准备)
    - [4.2 模型推理](#42-模型推理)

---


## 1. 模型简介
FastVLM 是苹果公司开发并开源的一系列视觉语言模型，它能够近乎即时地处理高分辨率图像并生成文字描述，而且能够分析图像中的文字、图表、图标、图形和布局。

本例程对FastVLM的0.5B(https://huggingface.co/apple/FastVLM-0.5B)和1.5B模型(https://huggingface.co/apple/FastVLM-1.5B)进行移植，可在EA65xx平台上进行图片+文字输入的多模态推理。

## 2. 特性
### 2.1 目录结构说明
项目目录结构如下：
```
├── scripts/     # 存放模型下载的shell脚本
└── README.md    # 本例程的中文指南
```

### 2.2 SDK特性
- 支持EA6530
- 支持图片+文字的多模态推理

## 3. 模型准备
### 3.1 使用提供的模型
可以通过以下命令下载我们编译好的模型：
```bash
chmod -R +x scripts
./scripts/download.sh
```
下载的模型文件包括：
```
├──/FastVLM-0.5B-Ai16Wpcqi8-2core
|   ├── analysis.json
|   ├── engine_strip.json
|   ├── llava_qwen2_strip.alm
|   ├── tokenizer_config.json
|   ├── vocab.json
|   ├── merges.txt
|   ├── panda_448x448.jpg
├──/FastVLM-1.5B-Ai16Wpcqi8-2core
|   ├── analysis.json
|   ├── engine_strip.json
|   ├── llava_qwen2_strip.alm
|   ├── tokenizer_config.json
|   ├── vocab.json
|   ├── merges.txt
|   ├── panda_448x448.jpg
```

### 3.2 自行编译模型
也可以自行对模型进行编译。
编译过程首先需要下载官方仓库中的源模型，之后需要使用taNNTC工具将源模型编译成alm文件才可在EA65xx平台上运行，并且为了将模型量化到Ai16Wpcqi8格式，需要准备一张图片。

#### 3.2.1 下载官方仓库源模型
在modelscope或者huggingface下载模型，这里以huggingface的0.5B模型为例：
```bash
huggingface-cli download apple/FastVLM-0.5B --local-dir ./FastVLM-0.5B
```

#### 3.2.2 准备taNNTC工具
具体过程可参考[taNNTC环境搭建](../../docs/环境安装指南.md)，如果未拉取最新的taNNTC镜像可能导致量化出错

#### 3.2.3 准备量化图片
准备一张分辨率为448x448的图片，如panda_448x448.jpg，并放在模型文件夹下，如./FastVLM-0.5B/panda_448x448.jpg

#### 3.2.4 执行编译过程
进入模型文件夹的上一层目录，执行下面命令(注意将panda_448x448.jpg改为实际准备的量化图片名)：
```bash
llm_build --model_path ./FastVLM-0.5B --model_filename FastVLM-0.5B-quant-Ai16Wpcqi8-1core --quantized_type Ai16Wpcqi8  --processes 4 --height 448 --width 448 --max_seq_len 640 --cache_size 640 --prompt "decribe the image" --image ./FastVLM-0.5B/panda_448x448.jpg --gen_real_input
```
如果需要配置双核联合模式，还需要加上 
```
--prefill_core_count 2 --decode_core_count 2
```

#### 3.2.5 对权重文件进行裁剪
上述命令执行完后，需要进入权重文件所在的文件夹，再对权重文件进行裁剪以节省内存：
```bash
almstrip engine.json
```

## 4. 模型使用
### 4.1 环境准备
针对EA65XX平台，SDK安装完成后，已经内置了taRuntime、taOpenCV等所需运行库，可直接运行。只需把第2步准备好的模型和测试图片拷贝到EA65XX平台即可。

### 4.2 模型推理
在模型文件夹中运行下面命令：
```bash
acuity-llm-image-cli engine_strip.json --p 1
```
等待模型加载完成以后就可以输入图片路径和问题来进行推理，其中--p设为1是用来跳过输出采样环节以加快推理。