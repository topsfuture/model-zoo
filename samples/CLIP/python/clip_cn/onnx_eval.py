#!/usr/bin/env python3
"""
CIFAR-100 零样本评估脚本
"""
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from bert_tokenizer import bert_tokenize

import cv2


CHINESE_TEMPLATES = [
    lambda c: f"{c}的照片",
    lambda c: f"质量差的{c}的照片",
    lambda c: f"许多{c}的照片",
    lambda c: f"{c}的雕塑",
    lambda c: f"难以看到{c}的照片",
    lambda c: f"{c}的低分辨率照片",
    lambda c: f"{c}的渲染",
    lambda c: f"涂鸦{c}",
    lambda c: f"{c}的糟糕照片",
    lambda c: f"{c}的裁剪照片",
    lambda c: f"{c}的纹身",
    lambda c: f"{c}的刺绣照片",
    lambda c: f"很难看到{c}的照片",
    lambda c: f"{c}的明亮照片",
    lambda c: f"一张干净的{c}的照片",
    lambda c: f"一张包含{c}的照片",
    lambda c: f"{c}的深色照片",
    lambda c: f"{c}的手绘画",
    lambda c: f"我的{c}的照片",
    lambda c: f"不自然的{c}的照片",
    lambda c: f"一张酷的{c}的照片",
    lambda c: f"{c}的特写照片",
    lambda c: f"{c}的黑白照片",
    lambda c: f"一幅{c}的画",
    lambda c: f"一幅{c}的绘画",
    lambda c: f"一张{c}的像素照片",
    lambda c: f"{c}的雕像",
    lambda c: f"一张{c}的明亮照片",
    lambda c: f"{c}的裁剪照片",
    lambda c: f"人造的{c}的照片",
    lambda c: f"一张关于{c}的照片",
    lambda c: f"损坏的{c}的jpeg照片",
    lambda c: f"{c}的模糊照片",
    lambda c: f"{c}的相片",
    lambda c: f"一张{c}的好照片",
    lambda c: f"{c}的渲染照",
    lambda c: f"视频游戏中的{c}",
    lambda c: f"一张{c}的照片",
    lambda c: f"{c}的涂鸦",
    lambda c: f"{c}的近距离照片",
    lambda c: f"{c}的折纸",
    lambda c: f"{c}在视频游戏中",
    lambda c: f"{c}的草图",
    lambda c: f"{c}的涂鸦照",
    lambda c: f"{c}的折纸形状",
    lambda c: f"低分辨率的{c}的照片",
    lambda c: f"玩具{c}",
    lambda c: f"{c}的副本",
    lambda c: f"{c}的干净的照片",
    lambda c: f"一张大{c}的照片",
    lambda c: f"{c}的重现",
    lambda c: f"一张漂亮的{c}的照片",
    lambda c: f"一张奇怪的{c}的照片",
    lambda c: f"模糊的{c}的照片",
    lambda c: f"卡通{c}",
    lambda c: f"{c}的艺术作品",
    lambda c: f"{c}的素描",
    lambda c: f"刺绣{c}",
    lambda c: f"{c}的像素照",
    lambda c: f"{c}的拍照",
    lambda c: f"{c}的损坏的照片",
    lambda c: f"高质量的{c}的照片",
    lambda c: f"毛绒玩具{c}",
    lambda c: f"漂亮的{c}的照片",
    lambda c: f"小{c}的照片",
    lambda c: f"照片是奇怪的{c}",
    lambda c: f"漫画{c}",
    lambda c: f"{c}的艺术照",
    lambda c: f"{c}的图形",
    lambda c: f"大{c}的照片",
    lambda c: f"黑白的{c}的照片",
    lambda c: f"{c}毛绒玩具",
    lambda c: f"一张{c}的深色照片",
    lambda c: f"{c}的摄影图",
    lambda c: f"{c}的涂鸦照",
    lambda c: f"玩具形状的{c}",
    lambda c: f"拍了{c}的照片",
    lambda c: f"酷酷的{c}的照片",
    lambda c: f"照片里的小{c}",
    lambda c: f"{c}的刺青",
    lambda c: f"{c}的可爱的照片",
    lambda c: f"一张{c}可爱的照片",
    lambda c: f"{c}可爱图片",
    lambda c: f"{c}酷炫图片",
    lambda c: f"一张{c}的酷炫的照片",
    lambda c: f"一张{c}的酷炫图片",
    lambda c: f"这是{c}",
    lambda c: f"{c}的好看照片",
    lambda c: f"一张{c}的好看的图片",
    lambda c: f"{c}的好看图片",
    lambda c: f"{c}的照片。",
    lambda c: f"质量差的{c}的照片。",
    lambda c: f"许多{c}的照片。",
    lambda c: f"{c}的雕塑。",
    lambda c: f"难以看到{c}的照片。",
    lambda c: f"{c}的低分辨率照片。",
    lambda c: f"{c}的渲染。",
    lambda c: f"涂鸦{c}。",
    lambda c: f"{c}的糟糕照片。",
    lambda c: f"{c}的裁剪照片。",
    lambda c: f"{c}的纹身。",
    lambda c: f"{c}的刺绣照片。",
    lambda c: f"很难看到{c}的照片。",
    lambda c: f"{c}的明亮照片。",
    lambda c: f"一张干净的{c}的照片。",
    lambda c: f"一张包含{c}的照片。",
    lambda c: f"{c}的深色照片。",
    lambda c: f"{c}的手绘画。",
    lambda c: f"我的{c}的照片。",
    lambda c: f"不自然的{c}的照片。",
    lambda c: f"一张酷的{c}的照片。",
    lambda c: f"{c}的特写照片。",
    lambda c: f"{c}的黑白照片。",
    lambda c: f"一幅{c}的画。",
    lambda c: f"一幅{c}的绘画。",
    lambda c: f"一张{c}的像素照片。",
    lambda c: f"{c}的雕像。",
    lambda c: f"一张{c}的明亮照片。",
    lambda c: f"{c}的裁剪照片。",
    lambda c: f"人造的{c}的照片。",
    lambda c: f"一张关于{c}的照片。",
    lambda c: f"损坏的{c}的jpeg照片。",
    lambda c: f"{c}的模糊照片。",
    lambda c: f"{c}的相片。",
    lambda c: f"一张{c}的好照片。",
    lambda c: f"{c}的渲染照。",
    lambda c: f"视频游戏中的{c}。",
    lambda c: f"一张{c}的照片。",
    lambda c: f"{c}的涂鸦。",
    lambda c: f"{c}的近距离照片。",
    lambda c: f"{c}的折纸。",
    lambda c: f"{c}在视频游戏中。",
    lambda c: f"{c}的草图。",
    lambda c: f"{c}的涂鸦照。",
    lambda c: f"{c}的折纸形状。",
    lambda c: f"低分辨率的{c}的照片。",
    lambda c: f"玩具{c}。",
    lambda c: f"{c}的副本。",
    lambda c: f"{c}的干净的照片。",
    lambda c: f"一张大{c}的照片。",
    lambda c: f"{c}的重现。",
    lambda c: f"一张漂亮的{c}的照片。",
    lambda c: f"一张奇怪的{c}的照片。",
    lambda c: f"模糊的{c}的照片。",
    lambda c: f"卡通{c}。",
    lambda c: f"{c}的艺术作品。",
    lambda c: f"{c}的素描。",
    lambda c: f"刺绣{c}。",
    lambda c: f"{c}的像素照。",
    lambda c: f"{c}的拍照。",
    lambda c: f"{c}的损坏的照片。",
    lambda c: f"高质量的{c}的照片。",
    lambda c: f"毛绒玩具{c}。",
    lambda c: f"漂亮的{c}的照片。",
    lambda c: f"小{c}的照片。",
    lambda c: f"照片是奇怪的{c}。",
    lambda c: f"漫画{c}。",
    lambda c: f"{c}的艺术照。",
    lambda c: f"{c}的图形。",
    lambda c: f"大{c}的照片。",
    lambda c: f"黑白的{c}的照片。",
    lambda c: f"{c}毛绒玩具。",
    lambda c: f"一张{c}的深色照片。",
    lambda c: f"{c}的摄影图。",
    lambda c: f"{c}的涂鸦照。",
    lambda c: f"玩具形状的{c}。",
    lambda c: f"拍了{c}的照片。",
    lambda c: f"酷酷的{c}的照片。",
    lambda c: f"照片里的小{c}。",
    lambda c: f"{c}的刺青。",
    lambda c: f"{c}的可爱的照片。",
    lambda c: f"一张{c}可爱的照片。",
    lambda c: f"{c}可爱图片。",
    lambda c: f"{c}酷炫图片。",
    lambda c: f"一张{c}的酷炫的照片。",
    lambda c: f"一张{c}的酷炫图片。",
    lambda c: f"这是{c}。",
    lambda c: f"{c}的好看照片。",
    lambda c: f"一张{c}的好看的图片。",
    lambda c: f"{c}的好看图片。",
    lambda c: f"一种叫{c}的花的照片",
    lambda c: f"一种叫{c}的食物的照片",
    lambda c: f"{c}的卫星照片"
]



def get_clip_transform_opencv(image_size=224):
    MEAN_R = 0.48145466
    MEAN_G = 0.4578275
    MEAN_B = 0.40821073
    STD_R = 0.26862954
    STD_G = 0.26130258
    STD_B = 0.27577711
    
    def transform_opencv(pil_image):
        """
        输出: torch.Tensor
        """
        # 1. 转换为numpy数组 (OpenCV格式: BGR)
        img_np = np.array(pil_image)
        
        # 2. PIL是RGB，OpenCV是BGR，需要转换
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 3. 调整大小（保持长宽比，最小边缩放到image_size）
        h, w = img_cv.shape[:2]
        new_h, new_w = h, w
        
        scale = float(image_size) / min(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        img_resized = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # 4. 中心裁剪到image_size x image_size
        start_h = (new_h - image_size) // 2
        start_w = (new_w - image_size) // 2
        
        if start_h < 0 or start_w < 0 or start_h + image_size > new_h or start_w + image_size > new_w:
            img_cropped = cv2.resize(img_resized, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        else:
            img_cropped = img_resized[start_h:start_h+image_size, start_w:start_w+image_size]
        
        # 5. 转换为float并归一化到[0, 1]
        img_float = img_cropped.astype(np.float32) / 255.0
        
        # 6. 应用归一化参数
        img_float[:, :, 0] = (img_float[:, :, 0] - MEAN_B) / STD_B  # B
        img_float[:, :, 1] = (img_float[:, :, 1] - MEAN_G) / STD_G  # G
        img_float[:, :, 2] = (img_float[:, :, 2] - MEAN_R) / STD_R  # R
        
        # 7. 转换为RGB顺序（PyTorch期望的是RGB）
        img_rgb = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
        
        # 8. 转换为PyTorch Tensor (HWC -> CHW)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()
        
        return img_tensor
    
    return transform_opencv




class LocalCIFAR100Dataset(Dataset):
    def __init__(self, data_root, split='test', transform=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        
        # 1. 加载中文标签
        label_file = os.path.join(data_root, 'label_cn.txt')
        with open(label_file, 'r', encoding='utf-8') as f:
            self.classnames = [line.strip() for line in f]
        self.num_classes = len(self.classnames)
        
        # 2. 按照数字顺序构建图片路径
        self.image_paths = []
        self.labels = []
        
        split_dir = os.path.join(data_root, split)
        
        # 遍历 000, 001, ..., 099
        for label_idx in range(self.num_classes):
            # 将标签索引转换为3位数字字符串
            class_folder = f"{label_idx:03d}"  # 0 -> "000", 1 -> "001", ...
            class_dir = os.path.join(split_dir, class_folder)
            
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(label_idx)
            else:
                print(f"警告: 类别文件夹不存在: {class_dir}")
        
        print(f"加载了 {len(self.image_paths)} 张图片，来自 {self.num_classes} 个类别")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    
    



def get_clip_transform(image_size=224):
    """CLIP 标准图像预处理"""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                           (0.26862954, 0.26130258, 0.27577711)),
    ])

def build_zero_shot_classifier(text_session, classnames, templates, context_length=52):
    """构建零样本分类器（文本特征矩阵）"""
    print(f"为 {len(classnames)} 个类别构建零样本分类器...")
    
    # 获取 ONNX 模型输入输出名称
    input_name = text_session.get_inputs()[0].name
    output_name = text_session.get_outputs()[0].name
    
    # 处理批处理问题
    classifier_weights = []
    
    for classname in tqdm(classnames, desc="处理类别"):
        texts = [template(classname) for template in templates]
        
        # 处理批次：由于ONNX模型只支持batch_size=1，需要逐个处理
        text_features_list = []
        for text in texts:
            # 为单个文本tokenize
            text_tokens = bert_tokenize([text], context_length)  # 注意：传入列表
            
            # 检查形状
            if text_tokens.shape != (1, context_length):
                print(f"警告: tokenize后的形状为 {text_tokens.shape}，期望 (1, {context_length})")
                continue
                
            # 运行文本模型推理
            try:
                text_feature = text_session.run([output_name], {input_name: text_tokens})[0]
                text_features_list.append(torch.from_numpy(text_feature))
            except Exception as e:
                print(f"处理文本 '{text[:20]}...' 时出错: {e}")
                continue
        
        if not text_features_list:
            print(f"警告: 类别 '{classname}' 没有生成有效的特征")
            continue
            
        # 合并特征
        text_features = torch.stack(text_features_list, dim=0)  # 形状: [num_templates, feature_dim]
        
        # 归一化
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        class_feature = text_features.mean(dim=0)
        class_feature = class_feature / class_feature.norm()
        
        # 确保 class_feature 是一维的
        if len(class_feature.shape) > 1:
            class_feature = class_feature.squeeze()
        
        classifier_weights.append(class_feature)
    
    # 调试信息
    print(f"分类器权重数量: {len(classifier_weights)}")
    if classifier_weights:
        print(f"第一个权重形状: {classifier_weights[0].shape}")
    
    classifier_matrix = torch.stack(classifier_weights, dim=1)  # 形状: [feature_dim, num_classes]
    print(f"分类器矩阵形状: {classifier_matrix.shape}")
    return classifier_matrix



def evaluate_on_cifar100(vision_session, classifier_matrix, dataloader, num_classes=100):
    """在 CIFAR-100 上评估模型"""
    print(f"开始在 {len(dataloader.dataset)} 张图片上评估...")
    
    # 获取 ONNX 模型输入输出名称
    input_name = vision_session.get_inputs()[0].name
    output_name = vision_session.get_outputs()[0].name
    
    # 检查分类器矩阵形状
    print(f"分类器矩阵形状: {classifier_matrix.shape}")
    expected_classes = classifier_matrix.shape[1]
    if expected_classes != num_classes:
        print(f"警告: 分类器矩阵有 {expected_classes} 个类别，但数据集有 {num_classes} 个类别")
    
    total_correct = 0
    total_samples = 0
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="评估批次")):
        # 准备图像输入
        images_np = images.numpy().astype(np.float32)
        batch_size = images_np.shape[0]
        
        # 运行视觉模型推理
        image_features = vision_session.run([output_name], {input_name: images_np})[0]
        image_features = torch.from_numpy(image_features)
        
        # 调试：打印第一个batch的特征形状
        if batch_idx == 0:
            print(f"第一个batch的图像特征形状: {image_features.shape}")
        
        # 检查并调整图像特征形状
        # 预期: image_features 应该是 [batch_size, feature_dim]
        # 但如果模型返回 [feature_dim, batch_size]，需要转置
        if image_features.dim() == 2:
            if image_features.shape[0] == batch_size and image_features.shape[1] == classifier_matrix.shape[0]:
                # 形状正确: [batch_size, feature_dim]
                pass
            elif image_features.shape[0] == classifier_matrix.shape[0] and image_features.shape[1] == batch_size:
                # 形状是 [feature_dim, batch_size]，需要转置
                image_features = image_features.t()
            else:
                print(f"警告: 图像特征形状异常: {image_features.shape}")
                # 尝试自动调整
                if image_features.shape[0] == batch_size:
                    # 假设是 [batch_size, feature_dim]
                    pass
                elif image_features.shape[1] == batch_size:
                    # 假设是 [feature_dim, batch_size]
                    image_features = image_features.t()
        
        # 归一化图像特征
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 计算相似度（logits）
        # 图像特征: [batch, feature_dim]
        # 分类器矩阵: [feature_dim, num_classes]
        # 相似度: [batch, num_classes]
        similarity = 100.0 * torch.matmul(image_features, classifier_matrix)
        
        # 获取预测结果
        predictions = similarity.argmax(dim=1)
        
        # 统计准确率
        correct = (predictions == labels).sum().item()
        total_correct += correct
        total_samples += batch_size
        
        # 统计每个类别的准确率
        for i in range(batch_size):
            label = labels[i].item()
            per_class_total[label] += 1
            if predictions[i].item() == label:
                per_class_correct[label] += 1
    
    # 计算总体准确率
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # 计算每个类别的准确率
    per_class_accuracy = []
    for i in range(num_classes):
        if per_class_total[i] > 0:
            acc = per_class_correct[i] / per_class_total[i]
            per_class_accuracy.append(acc)
        else:
            per_class_accuracy.append(0.0)
    
    return overall_accuracy, per_class_accuracy



def main():
    parser = argparse.ArgumentParser(description='独立 CIFAR-100 零样本评估脚本')
    
    # 必需参数
    parser.add_argument('--vision-onnx', type=str, required=True,
                       help='视觉 ONNX 模型路径 (如: vit-b-16.img.fp32.onnx)')
    parser.add_argument('--text-onnx', type=str, required=True,
                       help='文本 ONNX 模型路径 (如: vit-b-16.txt.fp32.onnx)')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='CIFAR-100 数据目录 (包含 label_cn.txt 和 test/ 子目录)')
    
    # 可选参数
    parser.add_argument('--image-size', type=int, default=224,
                       help='图像输入大小 (默认: 224)')
    parser.add_argument('--context-length', type=int, default=52,
                       help='文本上下文长度 (默认: 52)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='评估批次大小 (默认: 1)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载线程数 (默认: 4)')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'test'],
                       help='使用训练集还是测试集 (默认: test)')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='结果保存目录 (默认: ./results)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("独立 CIFAR-100 零样本评估")
    print("=" * 60)
    print(f"视觉模型: {args.vision_onnx}")
    print(f"文本模型: {args.text_onnx}")
    print(f"数据目录: {args.data_dir}")
    print(f"图像大小: {args.image_size}")
    print(f"批次大小: {args.batch_size}")
    print()
    
    # 1. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. 加载 ONNX 模型
    print("1. 加载 ONNX 模型...")
    providers = ['CPUExecutionProvider']
    vision_session = ort.InferenceSession(args.vision_onnx, providers=providers)
    text_session = ort.InferenceSession(args.text_onnx, providers=providers)
    
    # 检查模型输入输出
    vision_input_shape = vision_session.get_inputs()[0].shape
    text_input_shape = text_session.get_inputs()[0].shape
    print(f"  视觉模型输入形状: {vision_input_shape}")
    print(f"  文本模型输入形状: {text_input_shape}")
    
    # 3. 准备数据集
    print("\n2. 加载本地中文化 CIFAR-100 数据集...")
    transform = get_clip_transform_opencv(args.image_size)
    
    dataset = LocalCIFAR100Dataset(
        data_root=args.data_dir,
        split=args.split,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    # 4. 构建零样本分类器
    print("\n3. 构建零样本分类器...")
    classifier_matrix = build_zero_shot_classifier(
        text_session=text_session,
        classnames=dataset.classnames,
        templates=CHINESE_TEMPLATES,
        context_length=args.context_length
    )
    
    # 验证分类器矩阵
    if classifier_matrix.shape[1] != dataset.num_classes:
        print(f"错误: 分类器矩阵只有 {classifier_matrix.shape[1]} 个类别，但需要 {dataset.num_classes} 个类别")
        print("可能的原因:")
        print("1. 文本模型推理失败")
        print("2. tokenizer 出现问题")
        print("3. 模板生成失败")
        return
    
    # 5. 运行评估
    print("\n4. 开始评估...")
    overall_acc, per_class_acc = evaluate_on_cifar100(
        vision_session=vision_session,
        classifier_matrix=classifier_matrix,
        dataloader=dataloader,
        num_classes=dataset.num_classes
    )
    
    # 6. 保存结果
    print("\n5. 保存评估结果...")
    results = {
        'vision_model': args.vision_onnx,
        'text_model': args.text_onnx,
        'dataset': 'cifar-100',
        'split': args.split,
        'num_samples': len(dataset),
        'overall_accuracy': overall_acc,
        'per_class_accuracy': per_class_acc,
        'class_names': dataset.classnames
    }
    
    result_file = os.path.join(args.output_dir, f'cifar100_results_{args.split}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 7. 打印汇总结果
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    print(f"数据集: CIFAR-100 ({args.split}集)")
    print(f"样本数量: {len(dataset)}")
    print(f"类别数量: {dataset.num_classes}")
    print(f"总体准确率: {overall_acc:.4%}")
    print(f"总体准确率 (数值): {overall_acc:.6f}")
    
    # 打印每个类别的准确率
    print(f"\n每个类别的准确率:")
    for i, (classname, acc) in enumerate(zip(dataset.classnames, per_class_acc)):
        if i < 10:  # 只显示前10个类别
            print(f"  {classname}: {acc:.4%}")
    if len(dataset.classnames) > 10:
        print(f"  ... 和另外 {len(dataset.classnames)-10} 个类别")
    
    # 计算平均类别准确率
    mean_per_class_acc = np.mean(per_class_acc)
    print(f"平均类别准确率: {mean_per_class_acc:.4%}")
    
    print(f"\n详细结果已保存到: {result_file}")
    print("=" * 60)
    
    return overall_acc

if __name__ == '__main__':
    main()