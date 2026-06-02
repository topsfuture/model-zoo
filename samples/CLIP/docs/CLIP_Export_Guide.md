# CLIP 模型的导出
## 1. 导出 clip_en 模型

需要通过源码来导出 onnx 文件，clip的变种很多，但是思想类似，以 openai 原始仓库[CLIP官方开源仓库](https://github.com/openai/CLIP)为例。

### 导出encode_image部分

模型分encode_image和encode_text两部分，以ViT-B/32模型为例，如果需要导出encode_image部分，修改源码 CLIP/clip/model.py:358 forward()函数
```python
    # def forward(self, image, text):
    def forward(self, image):
        image_features = self.encode_image(image)
        # text_features = self.encode_text(text)

        # normalized features
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        # return logits_per_image, logits_per_text
        return image_features
```
然后运行以下代码导出onnx模型

```python
import torch
from clip import *
from PIL import Image
import torch

device =  "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) # 此处路径可以替换成本地pt文件路径
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog"] * 256).to(device)

with torch.no_grad():
    # Assuming 'model' is your PyTorch model and 'text' is the input tensor
    torch.onnx.export(
        model,                # model being run
        image,                 # model input (or a tuple for multiple inputs)
        "clip_image_vitb32.onnx",          # where to save the model (can be a file or file-like object)
        dynamic_axes={'image': {0: 'batch_size'},
                      'output': {0: 'batch_size'}},  # dynamic axes of the input
        input_names=['image'], # setting the input name to 'text'
        output_names=['output'] # you can also set the output name(s) if necessary
    )
```

### 导出encode_text部分

同理，修改源码 CLIP/clip/model.py:358处；
```python
    # def forward(self, image, text):
    def forward(self, text):
        # image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        # return logits_per_image, logits_per_text
        return text_features
```
另外，注意在CLIP/clip/model.py:354行`x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection`也需要注释掉，并同时保存导出onnx时的self.text_projection数据，为后续推理使用；
```python
    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)

        
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        np.save('text_projection_512_512.npy',self.text_projection)
        
        return x
```


然后运行以下代码导出onnx模型

```python
import torch
from clip import *
from PIL import Image
import torch

device =  "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) # 此处路径可以替换成本地pt文件路径
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog"] * 256).to(device)

with torch.no_grad():
    # Assuming 'model' is your PyTorch model and 'text' is the input tensor
    torch.onnx.export(
        model,                # model being run
        text,                 # model input (or a tuple for multiple inputs)
        "clip_text_vitb32.onnx",          # where to save the model (can be a file or file-like object)
        dynamic_axes={'text': {0: 'batch_size'},
                      'output': {0: 'batch_size'}},  # dynamic axes of the input
        input_names=['text'], # setting the input name to 'text'
        output_names=['output'] # you can also set the output name(s) if necessary
    )
```


## 2.CLIP_cn 模型的导出

本例程使用 Chinese-CLIP vit-b-16 预训练模型，该模型由达摩院基于 OpenCLIP 完成，前往达摩院的 github 仓库[chinese-CLIP开源仓库]https://github.com/OFA-Sys/Chinese-CLIP. 在该页面获取预训练模型 `clip_cn_vit-b-16.pt`，之后的部署中将 CLIP 分为encode_image和encode_text两个子模型。


### 环境准备

+ **CUDA**：推荐[CUDA](https://developer.nvidia.com/cuda-11-6-0-download-archive)版本11.6及以上
+ **CUDNN**：推荐[CUDNN](https://developer.nvidia.com/rdp/cudnn-archive) 8.6.0及以上
+ **ONNX**：本文以onnx版本1.13.0，onnxruntime-gpu版本1.13.1，onnxmltools版本1.11.1为例
+ **Pytorch**：推荐1.12.1及以上，本文以1.12.1为例（建议直接pip安装1.12.1+cu116，环境尽量不要再使用conda安装cudatoolkit，避免环境CUDNN版本变化，导致TensorRT报错）
+ 其他依赖项

### 导出
在上述仓库找到 [pt_2_onnx脚本](../../../github/Chinese-CLIP/cn_clip/deploy/pytorch_to_onnx.py)
```bash
cd Chinese-CLIP/
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

checkpoint_path=</pretrained_weights/clip_cn_vit-b-16.pt> # 占位符中替换为指定要转换的.pt 模型完整路径
mkdir -p ./deploy/ # 创建ONNX模型的输出文件夹

python cn_clip/deploy/pytorch_to_onnx.py \
       --model-arch ViT-B-16 \
       --pytorch-ckpt-path ${checkpoint_path} \
       --save-onnx-path ./deploy/vit-b-16 \
       --convert-text --convert-vision

```


运行此代码转换完成后，将得到以下的log输出：
```
Finished PyTorch to ONNX conversion...
>>> The text FP32 ONNX model is saved at ${DATAPATH}/deploy/vit-b-16.txt.fp32.onnx
>>> The text FP16 ONNX model is saved at ${DATAPATH}/deploy/vit-b-16.txt.fp16.onnx with extra file ${DATAPATH}/deploy/vit-b-16.txt.fp16.onnx.extra_file
>>> The vision FP32 ONNX model is saved at ${DATAPATH}/deploy/vit-b-16.img.fp32.onnx
>>> The vision FP16 ONNX model is saved at ${DATAPATH}/deploy/vit-b-16.img.fp16.onnx with extra file ${DATAPATH}/deploy/vit-b-16.img.fp16.onnx.extra_file
```

上面示例代码执行结束后，我们得到了ViT-B-16规模，Chinese-CLIP文本侧和图像侧的ONNX模型，可以分别用于提取图文特征。输出ONNX模型的路径均以运行脚本时的`save-onnx-path`为前缀，后面依次拼上`.img`/`.txt`、`.fp16`/`.fp32`、`.onnx`。

注意到部分ONNX模型文件还附带有一个extra_file，其也是对应ONNX模型的一部分。在使用这些ONNX模型时，由于`.onnx`文件存储了extra_file的路径（如`${DATAPATH}/deploy/vit-b-16.txt.fp16.onnx.extra_file`）并会根据此路径载入extra_file，所以使用ONNX模型请不要改动存放的路径名，转换时`${DATAPATH}`也尽量用相对路径（如`../datapath`），避免运行时按路径找不到extra_file报错

其中`vit-b-16.img.fp32.onnx` 和 `vit-b-16.txt.fp32.onnx` 是本例程要使用的模型，为项目方便可重命名为 `clip_cn_image.onnx` 和 `clip_cn_text.onnx`