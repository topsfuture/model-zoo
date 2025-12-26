# LPRNet模型导出
## 1.准备工作
LPRNet模型是基于[LPRNet开源仓库](https://github.com/sirius-ai/LPRNet_Pytorch)进行转换的，因此首先需要将该仓库clone到本地，并根据仓库要求安装配置Pytorch环境，确保该仓库所提供的模型能够正常推理运行。

同时需要安装onnx和onnxsim模块为后续模型转换提供对应的onnx环境。

## 2.导出onnx模型
如果使用taNNTC编译模型，则必须先将Pytorch模型导出为onnx模型。导出模型的方法可以参考[python/export_onnx.py](../python/export_onnx.py)文件。

这里需要注意要将上述提到的python文件放置到[LPRNet开源仓库](https://github.com/sirius-ai/LPRNet_Pytorch)的根目录下，然后执行脚本:
```bash
python3 export_onnx.py
```
即可生成所需要的lprnet.onnx文件