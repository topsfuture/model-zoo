
# Internvl3.5

## 目录
- [Internvl3.5](#internvl35)
  - [目录](#目录)
  - [1. 模型简介](#1-模型简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
  - [3. 模型准备](#3-模型准备)
    - [3.1 使用提供的模型](#31-使用提供的模型)
    - [3.2 自行编译模型](#32-自行编译模型)
      - [3.2.1 下载官方仓库源模型](#321-下载官方仓库源模型)
      - [3.2.2 准备量化图片](#322-准备量化图片)
      - [3.2.3 准备taNNTC工具](#323-准备tanntc工具)
      - [3.2.4 执行编译过程](#324-执行编译过程)
      - [3.2.5 对权重文件进行裁剪](#325-对权重文件进行裁剪)
      - [3.2.6 添加chat\_template.json文件](#326-添加chat_templatejson文件)
  - [4. 模型使用](#4-模型使用)
    - [4.1 环境准备](#41-环境准备)
    - [4.2 模型推理](#42-模型推理)

---


## 1. 模型简介
InternVL3.5是OpenGVLab推出的一系列先进的开源多模态大语言模型，显著提升了InternVL系列的通用性、推理能力和推理效率。

本例程对internvl3.5-2b(https://huggingface.co/OpenGVLab/InternVL3_5-2B) 模型进行移植，可在EA65xx平台上进行图片+文字输入的多模态推理。

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
下载的模型文件和测试图片包括：
```
├── analysis.json
├── engine_strip.json
├── internvl_chat_strip.alm
├── panda_448x448.jpg
├── red_panda_1000x747.jpg
├── tokenizer_config.json
├── vocab.json
├── chat_template.json
├── house_860x578.jpg
├── merges.txt
├── preprocessor_config.json
└── tokenizer.json
```

### 3.2 自行编译模型
也可以自行对模型进行编译。
编译过程首先需要下载官方仓库中的源模型，之后需要使用taNNTC工具将源模型编译成alm文件才可在EA65xx平台上运行，并且为了将模型量化到Ai16Wpcqi8格式，需要准备一张分辨率为448x448的图片。

#### 3.2.1 下载官方仓库源模型
```bash
hf download OpenGVLab/InternVL3_5-2B --local-dir ./internvl3.5-2b/model/
```

#### 3.2.2 准备量化图片
准备一张分辨率为448x448的图片，如panda_448x448.jpg，并放在模型文件夹下，如./internvl3.5-2b/panda_448x448.jpg

#### 3.2.3 准备taNNTC工具
具体过程可参考[taNNTC环境搭建](../../docs/环境安装指南.md)

#### 3.2.4 执行编译过程
进入模型文件夹(internvl3.5-2b/)，执行下面命令(注意将panda_448x448.jpg改为实际准备的量化图片名)：
```bash
llm_build --model_path ./model/ --model_filename test --quantized_type Ai16Wpcqi8  --processes 12 --height 448 --width 448 --max_seq_len 640 --cache_size 640 --prompt "decribe the image" --image ./panda_448x448.jpg --gen_real_input
```

#### 3.2.5 对权重文件进行裁剪
上述命令执行完后，需要进入权重文件所在的文件夹，再对权重文件进行裁剪以节省内存：
```bash
almstrip engine.json
```

#### 3.2.6 添加chat_template.json文件
还需要添加一个聊天模板文件，文件名设置为chat_template.json，文件内容如下：
```
{
    "chat_template": "internvl2_5"
}
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