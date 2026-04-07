
# Whisper

## 目录
- [Whisper](#whisper)
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
      - [3.2.3 准备量化文件](#323-准备量化文件)
      - [3.2.4 执行编译过程](#324-执行编译过程)
      - [3.2.5 对权重文件进行裁剪](#325-对权重文件进行裁剪)
  - [4. 模型使用](#4-模型使用)
    - [4.1 环境准备](#41-环境准备)
    - [4.2 模型推理](#42-模型推理)

---


## 1. 模型简介
Whisper 是一个开源的深度学习语音识别模型，由 OpenAI 开发，它能够实现实时、多语言的语音识别，并支持跨多种环境和设备的灵活部署。

本例程对whisper-small(https://huggingface.co/openai/whisper-small/tree/main) 模型进行移植，可在EA65xx平台上进行推理。

## 2. 特性
### 2.1 目录结构说明
项目目录结构如下：
```
├── scripts/     # 存放模型下载的shell脚本
└── README.md    # 本例程的中文指南
```

### 2.2 SDK特性
- 支持EA6530
- 支持语音识别(Automatic Speech Recognition)

## 3. 模型准备
### 3.1 使用提供的模型
可以通过以下命令下载我们编译好的模型：
```bash
chmod -R +x scripts
./scripts/download.sh
```
下载的模型文件和测试文件包括：
```
├── analysis.json
├── engine_strip.json
├── whisper_strip.alm
├── output.pcm
├── output.wav
├── tokenizer_config.json
├── vocab.json
├── merges.txt
├── preprocessor_config.json
└── tokenizer.json
```

### 3.2 自行编译模型
也可以自行对模型进行编译。
编译过程首先需要下载官方仓库中的源模型，之后需要使用taNNTC工具将源模型编译成alm文件才可在EA65xx平台上运行，并且为了将模型量化到Ai16Wpcqi8格式，需要准备一份wav音频文件。

#### 3.2.1 下载官方仓库源模型
在终端执行
```bash
hf download openai/whisper-small --local-dir ./whisper-small    
```

#### 3.2.2 准备taNNTC工具
具体过程可参考[taNNTC环境搭建](../../docs/环境安装指南.md)，如果未拉取最新的taNNTC镜像可能导致量化出错

#### 3.2.3 准备量化文件
准备一份wav音频文件，如output.wav，并放在模型文件夹下

#### 3.2.4 执行编译过程
进入模型文件夹，执行下面命令：
```bash
llm_build --model_path model_dir --model_filename build_dir --quantized_type Ai16Wpcqi8 --processes 2 --max_seq_len 256 --cache_size 512 --wav output.wav --gen_real_input --prefill_core_count 1 --decode_core_count 1
```
将其中的prefill_core_count和decode_core_count设置成2，将使用双核组合模式
#### 3.2.5 对权重文件进行裁剪
上述命令执行完后，需要进入权重文件所在的文件夹，再对权重文件进行裁剪以节省内存：
```bash
almstrip engine.json
```

## 4. 模型使用
### 4.1 环境准备
针对EA65XX平台，SDK安装完成后，已经内置了taRuntime、taOpenCV等所需运行库，可直接运行。只需把第2步准备好的模型和pcm音频文件拷贝到EA65XX平台即可。

### 4.2 模型推理
在模型文件夹中运行下面命令：
```bash
acuity-llm-asr-cli engine.json output.pcm
```
模型将识别出output.pcm音频中的文字并输出