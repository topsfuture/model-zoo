import torch
from retinaface import RetinaFace

cfg= {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

device = "cuda" if torch.cuda.is_available() else "cpu"

model = RetinaFace(cfg).to(device)
input_tensor = torch.rand((1, 3, 640, 640), dtype=torch.float32)
state_dict = torch.load("../model/mobilenet0.25_Final.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)

model.eval()

torch.onnx.export(
    model,
    input_tensor,
    "retinaface.onnx",
    input_names=["images"],
    output_names=["cls", "loc", "landmark"],
    opset_version=17,
)