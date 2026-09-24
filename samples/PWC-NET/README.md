# PWC-Net

## 目录

- [PWC-Net](#pwc-net)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK 特性](#22-sdk-特性)
  - [3. 数据准备与模型编译](#3-数据准备与模型编译)
    - [3.1 数据准备](#31-数据准备)
    - [3.2 从 checkpoint 导出 ONNX](#32-从-checkpoint-导出-onnx)
    - [3.3 ONNX 转换为 NBG](#33-onnx-转换为-nbg)
  - [4. 例程测试](#4-例程测试)
    - [4.1 环境准备](#41-环境准备)
    - [4.2 交叉编译](#42-交叉编译)
    - [4.3 部署到 EA6530](#43-部署到-ea6530)
      - [4.3.1 Host 端准备 MPI Sintel 数据集和 manifest](#431-host-端准备-mpi-sintel-数据集和-manifest)
      - [4.3.2 上传板端数据和 manifest](#432-上传板端数据和-manifest)
    - [4.4 图像测试](#44-图像测试)
    - [4.5 视频测试](#45-视频测试)
    - [4.6 MPI Sintel 精度评估](#46-mpi-sintel-精度评估)
      - [4.6.1 运行评测](#461-运行评测)
    - [4.7 性能与精度基准](#47-性能与精度基准)

## 1. 简介

本例程可在 EA6530 平台上进行 PWC-Net 光流推理，支持两张图片输入和视频输入。工程同时提供基于 MPI Sintel Clean 子集的精度和速度评估程序。

PWC-Net 的原始 PyTorch 模型使用 CUDA correlation 扩展。本例程通过 export_pwc_onnx.py 将 correlation 替换为可导出到 ONNX 的实现，再使用 convert_model build 转换为可在 EA6530 上运行的 NBG。

本例程包含：

- PyTorch checkpoint 到 ONNX 的导出脚本；
- 384×256 和 960×512 两种输入尺寸的 ONNX、NBG 和转换配置；
- 基于 TA Runtime 和 NBG linker 的 C++ 推理工程；
- PWC-Net 图片、视频测试程序；
- MPI Sintel manifest 生成、板端部署和精度评估工具。

## 2. 特性

### 2.1 目录结构说明

项目目录结构如下：

```text
PWC-NET/
├── cpp/                         # C++ 推理和评估程序
│   ├── CMakeLists.txt
│   ├── pwc_net_imgpair.cpp/.h  # 两张图片 -> PWC-Net .flo
│   ├── pwc_net_video.cpp       # 视频 -> 光流速度
│   └── pwc_net_eval.cpp        # .flo 精度评估
├── configs/                    # convert_model 配置
│   ├── pwc_net_config_fp16_384x256.json
│   └── pwc_net_config_fp16_960x512.json
├── python/                     # Host 端 Python 脚本和模型定义
│   ├── export_pwc_onnx.py
│   ├── evaluate_sintel_host.py
│   └── source/PyTorch/models/PWCNet.py
├── scripts/                    # 下载、数据集和板端测试脚本
│   ├── download.sh
│   ├── run_sintel_eval.sh
│   ├── make_sintel_manifests.py
│   └── prepare_sintel_board_subset.py
└── README.md
```

### 2.2 SDK 特性

- 支持 EA6530 平台；
- 支持 PWC-Net FP16 NBG；
- 支持 384×256 和 960×512 两种静态输入尺寸；
- C++ 运行时通过 ta_runtime_query 查询 NBG 输入、输出尺寸，不需要在代码中手动修改网络尺寸；
- 支持图片对、视频和 MPI Sintel 批量评测；
- 图片程序的 profiler 输出写入 stderr，不会污染 stdout 中的 .flo 二进制数据。

## 3. 数据准备与模型编译

### 3.1 数据准备

本例程在 scripts 目录下提供了模型和 MPI Sintel 数据集的下载脚本 download.sh。如果您希望自行准备模型和数据集，可跳过本小节，直接参考 3.2 从 checkpoint 导出 ONNX 进行模型转换。以下命令均从 PWC-NET 样例目录执行。

```bash
cd <model-zoo>/samples/PWC-NET
chmod -R +x scripts
./scripts/download.sh
```

下载脚本使用阿里云网盘分享链接自动获取文件 ID、下载并解压模型和数据。

下载的模型包括：

```text
models/
├── pwc_net.pth.tar
├── pwc_net_384x256.onnx
├── pwc_net_384x256_float16.nb
├── pwc_net_960x512.onnx
└── pwc_net_960x512_float16.nb
```

模型压缩包根目录应包含 models/ 目录。

下载的数据包括：

```text
datasets/
└── sintel/
    └── training/
        ├── clean/<scene>/frame_XXXX.png
        └── flow/<scene>/frame_XXXX.flo
```

其中clean 文件夹下为数据集图片，frame_0001.png, frame_0002.png,...是连续的帧导出图片。

flow 文件夹下为真实光流值(GT, Ground Truth)，作为精度测试时使用。

### 3.2 从 checkpoint 导出 ONNX

原始 PWC-Net 使用 CUDA correlation 扩展，不能直接导出为通用 ONNX。本工程的导出脚本会注入固定搜索范围的 CPU/ONNX correlation 实现，并替换为可在 CPU 上导出的 warp 实现。

Host 端 Python 环境需要安装：

```bash
pip install onnx==1.19.1 torch==2.8.0 numpy==2.3.3
```

在 Host 端执行：

```bash
cd <model-zoo>/samples/PWC-NET

python3 python/export_pwc_onnx.py \
  --checkpoint models/pwc_net.pth.tar \
  --output models/pwc_net_384x256.onnx \
  --height 256 --width 384

python3 python/export_pwc_onnx.py \
  --checkpoint models/pwc_net.pth.tar \
  --output models/pwc_net_960x512.onnx \
  --height 512 --width 960
```

执行时还会生成.data文件，此文件在后续中不需要可以忽略
导出的 Tensor 尺寸如下：

| 模型 | 输入 Tensor | 输出 Tensor |
|---|---|---|
| 384×256 | [1,6,256,384] | [1,2,64,96] |
| 960×512 | [1,6,512,960] | [1,2,128,240] |

### 3.3 ONNX 转换为 NBG

如果不重新编译模型，可以直接使用 models/ 目录中已经提供的 NBG。需要重新生成 NBG 时，使用 ta-nntc 容器。

#### 3.3.1 获取并启动 taNNTC 容器

convert_model 已经安装在 taNNTC Docker 镜像中。Docker 环境的完整安装说明请参考[taNNTC 环境搭建](../../docs/环境安装指南.md#1-tanntc环境搭建)。

启动容器，并将当前 PWC-NET 样例目录挂载到容器中的 `/workspace`：

```bash
docker run --privileged \
  --name ta-nntc \
  --network host \
  -d \
  -v "$(pwd)":/workspace \
  swr.cn-north-4.myhuaweicloud.com/topsfuture/ta-nntc-docker:latest \
  tail -f /dev/null
```

检查容器状态：

```bash
docker ps --filter name=ta-nntc
docker exec ta-nntc which convert_model
```

下面的转换命令假设当前工作目录就是 PWC-NET 样例根目录；容器内对应路径为 `/workspace`。

#### 3.3.2 生成 960×512 NBG

convert_model build 已经包含 ONNX 导出、模型转换和 NBG 生成，不需要在 build 后再次执行单独的 convert_model export。

```bash
docker exec ta-nntc bash -lc \
  'cd /workspace && \
   python3 /usr/local/convert_model_tool/convert_model.py build \
   --config configs/pwc_net_config_fp16_960x512.json \
   --output_dir acuity_960x512'
```

生成的 NBG 位于转换输出目录中，确认后可复制到 models/：

```bash
cp acuity_960x512/pwc_net_960x512_float16.nb models/
```

#### 3.3.3 生成 384×256 NBG

```bash
docker exec ta-nntc bash -lc \
  'cd /workspace && \
   python3 /usr/local/convert_model_tool/convert_model.py build \
   --config configs/pwc_net_config_fp16_384x256.json \
   --output_dir acuity_384x256'
```

生成的 NBG 位于 acuity_384x256/，确认后复制到 models/：

```bash
cp acuity_384x256/pwc_net_384x256_float16.nb models/
```

两个配置都是 FP16 配置，不包含 INT8 校准结果。

## 4. 例程测试

### 4.1 环境准备

EA6530 刷机后的系统已经预装 taRuntime、taOpenCV 等运行库，无需额外安装，可以直接作为程序运行环境。通常需要一台 x86 Host 作为开发环境，用于交叉编译 C++ 程序。

### 4.2 交叉编译

在 x86 Host 上使用 TACO SDK 搭建交叉编译环境，具体请参考[交叉编译环境搭建](../../docs/环境安装指南.md#21-交叉编译环境搭建)。获取 TACO SDK package 请联系项目支持团队。

进入解压后的 TACO SDK 目录，启动预配置好的 Docker 编译环境：

```bash
cd <TACO_SDK目录>
./start_workshop_docker.sh
```

启动后通过 docker ps 查看实际容器名。下面以 ta-workshop 为例；如果容器名不同，请替换命令中的容器名。请确保样例所在的 model-zoo 目录在容器中映射为 /workspace。

```bash
docker ps

docker exec ta-workshop bash -lc \
  'export PATH="/toolchain/bin:$PATH" && \
   cd /workspace/samples/PWC-NET/cpp && \
   rm -rf build && mkdir build && cd build && \
   cmake .. && make -j$(nproc)'
```

编译完成后，程序位于 cpp/build/：

| C++ 程序 | 说明 |
|---|---|
| pwc_net_imgpair | 两张图片输入，输出 PWC-Net .flo |
| pwc_net_video | 输入视频，输出每帧光流速度 |
| pwc_net_eval | 读取用户预测和真实光流值(GT) .flo，计算精度指标 |

### 4.3 部署到 EA6530

示例将程序、960×512 NBG 和板端测试脚本复制到板端：

```bash
# 按实际板端环境修改以下两个变量
BOARD=<用户名>@<EA6530_IP>
REMOTE=<EA6530上的工作目录>

scp cpp/build/{pwc_net_video,pwc_net_imgpair,pwc_net_eval} \
  "$BOARD:$REMOTE/"
scp models/pwc_net_960x512_float16.nb "$BOARD:$REMOTE/"
scp scripts/run_sintel_eval.sh "$BOARD:$REMOTE/"
```

板端 MPI Sintel 评测还需要图片、GT 光流和 manifest。数据集准备和上传应在部署阶段完成。

#### 4.3.1 Host 端准备 MPI Sintel 数据集和 manifest

归档不包含 MPI Sintel 原始数据集。如果通过 `scripts/download.sh` 完成了数据集下载，你将看到如下结构：

```text
datasets/sintel/training/
├── clean/<scene>/frame_XXXX.png
└── flow/<scene>/frame_XXXX.flo
```

在 Host 端进入 PWC-NET 样例目录，运行：

```bash
python3 scripts/make_sintel_manifests.py
python3 scripts/prepare_sintel_board_subset.py
```

make_sintel_manifests.py 扫描完整 Sintel 数据集，生成 clean 的 Host manifest 和板端路径 manifest。

prepare_sintel_board_subset.py 将图片和flo文件复制到 datasets/sintel_board_subset/training/，并生成约 115 对的板端 manifest：

```text
datasets/sintel_clean_board_subset_manifest.txt
```

#### 4.3.2 上传板端数据和 manifest

板端 manifest 中的路径已经是 /tmp/sintel_eval/training/...。上传时需要让目录结构与 manifest 保持一致：

```bash
# 按实际板端环境修改为自己的登录地址和工作目录
BOARD=<用户名>@<EA6530_IP>
REMOTE=<EA6530上的工作目录>

ssh "$BOARD" 'mkdir -p /tmp/sintel_eval'
scp -r datasets/sintel_board_subset/training \
  "$BOARD:/tmp/sintel_eval/"
scp datasets/sintel_clean_board_subset_manifest.txt \
  "$BOARD:/tmp/sintel_eval/manifest_imgpair.txt"
```

上传后检查：

```bash
ssh "$BOARD" 'wc -l /tmp/sintel_eval/manifest_imgpair.txt; \
  head -n 1 /tmp/sintel_eval/manifest_imgpair.txt; \
  first=$(awk "NR==1 {print \$1}" /tmp/sintel_eval/manifest_imgpair.txt); \
  test -f "$first"'
```

### 4.4 图像测试

#### 4.4.1 PWC-Net 图片测试

pwc_net_imgpair 接受两张图片和一个 NBG，输出 .flo。程序通过 ta_runtime_query 查询 NBG 的输入、输出尺寸，不需要手动修改 C++ 常量。`<测试图片1>`应当为`<测试图片2>` 的前一帧图片，此时光流模型来预测从图1到图2的光流向量。

```bash
cd <EA6530上的工作目录>

./pwc_net_imgpair \
  --model pwc_net_960x512_float16.nb \
  <测试图片1>.png <测试图片2>.png \
  -o <输出文件名>.flo
```

纯 NBG 耗时和端到端耗时等 profiler 信息写入 stderr。

#### 4.4.2 单个图对精度评估

```bash
./pwc_net_eval \
  --pred <用户预测>.flo \
  --gt <真实光流值>.flo
```

用户预测和真实光流值(用GT指代, 即Ground Truth)的尺寸不一致时，pwc_net_eval 保持预测用户预测尺寸，将 真实光流值 降采样到预测尺寸，并同步缩放 真实光流值 的水平、垂直位移分量。

### 4.5 视频测试

pwc_net_video 输入视频，逐帧调用 PWC-Net，并输出每帧的光流计算和整体耗时信息。

```bash
./pwc_net_video \
  --model pwc_net_960x512_float16.nb \
  --video <视频路径>.mp4 \
  --max-frames 30
```

### 4.6 MPI Sintel 精度评估

本节假设已经按照 4.3 完成板端图片、GT 光流和 manifest 的准备。板端一键评测脚本为 scripts/run_sintel_eval.sh，它读取的 manifest 路径固定为：

```text
/tmp/sintel_eval/manifest_imgpair.txt
```

#### 4.6.1 运行评测

```bash
cd <EA6530上的工作目录>
./run_sintel_eval.sh --model pwc_net_960x512_float16.nb
```

评测流程如下：

1. 读取 /tmp/sintel_eval/manifest_imgpair.txt；
2. 对每一对图片调用 pwc_net_imgpair；
3. 将预测结果写入 sintel_preds/*.flo；
4. 自动生成预测与 GT 的 eval_manifest.txt；
5. 调用 pwc_net_eval 统计精度；
6. 统计端到端耗时和纯 NBG 耗时。

当前脚本没有 --manifest 参数。如果使用其他位置的 manifest，需要将文件复制或软链接到 /tmp/sintel_eval/manifest_imgpair.txt，或者修改脚本中的 MANIFEST 变量。

评测结果文件位于：

```text
sintel_results/final_summary.txt
sintel_results/eval_summary.csv
sintel_results/pwc_timing.csv
sintel_results/pwc_nbg_timing.csv
```

PWC-Net 960×512 NBG 的输出为 240×128。当前评估保持预测原生尺寸，将 MPI Sintel GT 从 1024×436 降采样到 240×128，同时按尺寸比例缩放 GT 光流分量，不对预测结果做 upscale。

### 4.7 性能与精度基准

以下数据于 2026-09-14 在 EA6530 板端使用同一份 115 对 MPI Sintel Clean manifest 实测。

| 方法 | 纯计算平均耗时 | 端到端平均耗时 | Mean EPE | Angular | Bad3 | Bad5 |
|---|---:|---:|---:|---:|---:|---:|
| PWC-Net 960×512 | 282.699 ms（NBG） | 1205.348 ms/对 | 10.663496 | 33.108951° | 47.235677% | 36.686962% |

指标说明：

- Mean EPE：平均端点误差，预测光流向量与 GT 向量的欧氏距离；
- Angular Error：预测向量与 GT 向量的平均角度误差；
- Bad3：端点误差大于 3 像素的比例；
- Bad5：端点误差大于 5 像素的比例；
- PWC-Net 纯计算时间是 115 对中 pwc_nbg_ms 的平均值；
- 端到端时间包含读图、解码、预处理、光流计算和 .flo 写出。

