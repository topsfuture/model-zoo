"""
需将该文件放置于改仓库(https://github.com/sirius-ai/LPRNet_Pytorch)根目录下运行
"""
import onnx
import torch
from onnxsim import simplify
from model.LPRNet import build_lprnet
import os

def export_onnx(pretrained_model_path, img_size=[94,24], onnx_path="LPRNet.onnx", use_cuda=False):

    device = torch.device("cuda:0" if use_cuda else "cpu")

    lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=68, dropout_rate=0)
    lprnet.to(device)
    lprnet.eval()

    lprnet.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))

    input = torch.randn(1, 3, img_size[1], img_size[0]).to(device)

    # ===== 导出 ONNX =====
    torch.onnx.export(
        lprnet,
        input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
    )

    # ===== simplify =====
    model = onnx.load(onnx_path)
    model_simplified, check = simplify(model)

    if check:
        onnx.save(model_simplified, "lprnet.onnx")
        print("Simplified ONNX saved as lprnet.onnx")
    else:
        onnx.save(model, "lprnet.onnx")
        print("Simplification failed")
    
    # Delete intermediate files
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
        print(f"Deleted {onnx_path}")

    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)
        print(f"Deleted {data_path}")

if __name__ == "__main__":
    export_onnx(pretrained_model_path="./weights/Final_LPRNet_model.pth")