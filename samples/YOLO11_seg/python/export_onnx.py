from ultralytics import YOLO
import os

# Load model (try relative path first, then absolute)
model_path = "../models/yolo11s-seg.pt"
if not os.path.exists(model_path):
    model_path = "../models/yolo11s_seg.pt"

if not os.path.exists(model_path):
    print(f"Model not found. Please download yolo11s-seg.pt to ../models/")
    exit(1)

print(f"Loading model from {model_path}")
model = YOLO(model_path)

# Export to ONNX
print("Exporting to ONNX (opset=14, 640x640)...")
model.export(format="onnx", opset=14, imgsz=640)

print(f"Exported to {model_path.replace('.pt', '.onnx')}")
