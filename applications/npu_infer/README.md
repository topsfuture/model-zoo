

# 1. 性能测试
## 1.1 使用方法

```bash
Usage: ./npu_infer -m model.nb [-l loop_count] [-c core]
       ./npu_infer -m model1.nb model2.nb [-l loop_count]
       ./npu_infer --context_dir <dir> [-l loop_count]

Options:
  -m: Model file path(s)
  -l: Loop count for inference (default: 1)
  -c: Core selection (only valid for single model)
      0 - Run on core 0 only
      1 - Run on core 1 only
      2 - Run on both cores in parallel (default for single model)
  --context_dir, -d: Directory containing model.nb, input_*.tensor, output_*.tensor, ...
                     Mutually exclusive with -m.

Examples:
  Single model on both cores:  ./npu_infer -m model.nb -l 100
  With verification:           ./npu_infer --context_dir ./test_data -l 100

```

# 2. NPU 推理结果和工具链参考输出对比
## 2.1 使用方法
将 npu 推理结果和工具链参考输出对比,使用方法：
```bash
./npu_infer --context_dir <dir>
```
使用对比功能之前需要准备 context_dir 目录如下：
```bash
<context_dir>/
├── <model>.nb          # 目录下仅能存在一个 .nb 文件。
├── input_0.tensor      # 第0个输入
├── input_1.tensor      # 第1个输入（如果有）
├── output_0.tensor     # 第0个输出参考
├── output_1.tensor     # 第1个输出参考（如果有）
└── output_2.tensor     # 第2个输出参考（如果有）
```
注意：--context_dir 参数与 -m 参数不能同时使用。

## 2.2 参考 tensor 的获取方法
在 nntc 工具链（需要20260701以后ta-nntc-docker版本）环境中，导出 .nb 的模型同时，使用 --ref_data 参数获取输入输出参考 tensor，例如来说：
```bash
convert_model export --config <config.json> --ref_data
```
或者
```bash
convert_model build --config <config.json> --ref_data
```

tensor 文件将保存在 ref_data/ 目录下



# 3. npu_infer 工具编译方法
在 workshop docker 中，进入包含 npu_infer.c 的目录，执行编译命令：
```bash
riscv64-unknown-linux-gnu-gcc \
    -o npu_infer \
    npu_infer.c \
    -I/tps-future/ta-base/ta-runtime/include \
    -I/tps-future/ta-vsp/ta-unify-9200O/include \
    -I/tps-future/ta-vsp/ta-unify-9200O/include/ovxlib-package-dev \
    -I/tps-future/ta-vsp/ta-unify-9200O/include/Vivante_ML_Toolkit_OVXLIB_dev \
    -I/tps-future/ta-vsp/ta-sys/include \
    -I/tps-future/ta-vsp/others \
    -L/tps-future/ta-base/ta-runtime/lib \
    -L/tps-future/ta-vsp/ta-unify-9200O/lib \
    -L/tps-future/ta-vsp/ta-viplite-9200o/lib \
    -lta_runtime \
    -lVIPhal \
    -lNBGlinker \
    -lpthread \
    -lm \
    -O3

```
或者使用 makefile，进入包含 npu_infer.c 和 makefile 的目录，执行命令：

```bash
make
```


执行获得 npu_infer 可执行文件。




