#!/usr/bin/env python3
"""
生成 MBF 网络量化的输入数据
"""

import os
import cv2
import numpy as np
import sys
import json
from pathlib import Path
from tqdm import tqdm
import random

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入您已实现的推理模块
try:
    from inference import BuffaloScPredictor, create_predictor
    print("成功导入 inference.py 中的推理模块")
except ImportError as e:
    print(f"无法导入 inference.py: {e}")
    print("请确保当前目录包含 inference.py 文件")
    sys.exit(1)

def collect_image_paths(data_dir, max_images=100):
    """
    收集数据集中的图片路径
    
    Args:
        data_dir: 图片目录
        max_images: 最大图片数量
        
    Returns:
        list: 图片路径列表
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')
    image_paths = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if any(file.endswith(ext) for ext in image_extensions):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
                
                if len(image_paths) >= max_images:
                    return image_paths
    
    return image_paths

def generate_quantization_data_with_detector(data_dir, output_dir, num_samples=10, seed=42):
    """
    使用真实检测器生成量化数据
    
    Args:
        data_dir: 包含图片的目录
        output_dir: 输出目录
        num_samples: 需要生成的样本数量
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化预测器 (使用您的完整实现)
    print("1. 初始化 buffalo_sc 模型 (包含 det_500m.onnx 检测器)...")
    try:
        predictor = create_predictor(device='cpu')
        print("模型初始化成功")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return None
    
    # 收集图片路径
    print(f"\n2. 从 {data_dir} 收集图片...")
    all_image_paths = collect_image_paths(data_dir, max_images=num_samples * 3)  # 多收集一些，以防检测失败
    
    if not all_image_paths:
        print(f"在 {data_dir} 中未找到任何图片文件")
        return None
    
    print(f"   找到 {len(all_image_paths)} 张图片")
    
    # 随机打乱顺序
    random.shuffle(all_image_paths)
    
    # 生成样本
    print(f"\n3. 处理图片并生成 {num_samples} 个量化样本...")
    samples_generated = 0
    processed_info = []
    failed_images = []
    
    # 使用极低的检测阈值以确保能检测到人脸
    DETECTION_THRESHOLD = 0.01
    
    for img_idx, img_path in enumerate(tqdm(all_image_paths, desc="处理图片")):
        if samples_generated >= num_samples:
            break
        
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            failed_images.append(os.path.basename(img_path))
            continue
        
        try:
            faces = predictor.get(img, score_threshold=DETECTION_THRESHOLD, max_faces=1)
        except Exception as e:
            failed_images.append(f"{os.path.basename(img_path)}: {str(e)}")
            continue
        
        if not faces:
            failed_images.append(f"{os.path.basename(img_path)}: 未检测到人脸")
            continue
        
        # 取第一个检测到的人脸
        face_info = faces[0]
        
        # 从检测结果中获取边界框
        bbox = face_info['bbox']
        x1, y1, x2, y2 = bbox
        
        # 确保坐标有效
        height, width = img.shape[:2]
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        if x2 <= x1 or y2 <= y1:
            failed_images.append(f"{os.path.basename(img_path)}: 无效的边界框")
            continue
        
        # 裁剪人脸区域
        face_region = img[y1:y2, x1:x2]
        if face_region.size == 0:
            failed_images.append(f"{os.path.basename(img_path)}: 裁剪区域为空")
            continue
        
        # 1. 调整到112x112 (模拟对齐/裁剪后的处理)
        aligned_face = cv2.resize(face_region, (112, 112))
        
        try:
            # 2. 转换为RGB
            face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            
            # 3. 调整维度顺序: HWC -> CHW
            face_chw = np.transpose(face_rgb, (2, 0, 1))
            
            # 4. 添加批次维度: CHW -> NCHW
            face_nchw = np.expand_dims(face_chw, axis=0).astype(np.float32)
            
            # 5. 归一化
            processed_face = (face_nchw - 127.5) / 128.0
            
        except Exception as e:
            failed_images.append(f"{os.path.basename(img_path)}: 预处理失败 - {e}")
            continue
        
        # 验证输出形状
        if processed_face.shape != (1, 3, 112, 112):
            failed_images.append(f"{os.path.basename(img_path)}: 形状错误 {processed_face.shape}")
            continue
        
        # 保存为 .npy 文件
        sample_id = samples_generated
        npy_path = output_dir / f"mbf_input_{sample_id:03d}.npy"
        np.save(npy_path, processed_face)
        
        
        # 记录处理信息
        info = {
            'sample_id': sample_id,
            'source_image': os.path.basename(img_path),
            'bbox': bbox,
            'score': face_info['score'],
            'npy_file': npy_path.name,
            'shape': list(processed_face.shape),
            'data_range': [float(processed_face.min()), float(processed_face.max())],
            'data_mean': float(processed_face.mean()),
            'data_std': float(processed_face.std())
        }
        processed_info.append(info)
        
        samples_generated += 1
        
        # 打印进度
        if samples_generated % 5 == 0:
            print(f"   已生成 {samples_generated}/{num_samples} 个样本")
    
    # 生成元数据文件
    metadata = {
        'model': 'w600k_mbf.onnx',
        'detector': 'det_500m.onnx',
        'input_shape': [1, 3, 112, 112],
        'data_format': 'NCHW',
        'color_space': 'RGB',
        'normalization': '(img - 127.5) / 128.0',
        'detection_threshold': DETECTION_THRESHOLD,
        'samples_generated': samples_generated,
        'samples_requested': num_samples,
        'preprocessing_steps': [
            '1. 使用 det_500m.onnx 检测人脸',
            '2. 裁剪检测到的人脸区域',
            '3. 调整大小到 112x112',
            '4. BGR 转 RGB',
            '5. 维度转换: HWC -> CHW',
            '6. 添加批次维度: CHW -> NCHW',
            '7. 归一化: (img - 127.5) / 128.0'
        ],
        'generated_files': {
            'npy_files': f'mbf_input_*.npy (共 {samples_generated} 个)',
            'visualization': f'mbf_vis_*.jpg (共 {samples_generated} 个)',
            'aligned_faces': f'mbf_aligned_*.jpg (共 {samples_generated} 个)',
            'metadata': 'quant_metadata.json'
        },
        'failed_images': failed_images[:20]  # 只记录前20个失败
    }
    
    metadata_path = output_dir / "quant_metadata.json"
    # with open(metadata_path, 'w', encoding='utf-8') as f:
    #     json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # 创建合并的 .npy 文件 
    if samples_generated > 0:
        all_data = []
        for i in range(samples_generated):
            npy_file = output_dir / f"mbf_input_{i:03d}.npy"
            data = np.load(npy_file)
            all_data.append(data)
        
        all_data_array = np.concatenate(all_data, axis=0)  # 形状: (N, 3, 112, 112)
        combined_path = output_dir / "mbf_quant_all.npy"
        np.save(combined_path, all_data_array)
        
        metadata['generated_files']['combined_data'] = 'mbf_quant_all.npy'
        
        # 更新元数据
        # with open(metadata_path, 'w', encoding='utf-8') as f:
        #     json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("量化数据生成完成!")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")
    print(f"成功生成: {samples_generated}/{num_samples} 个样本")
    print(f"数据类型: float32")
    print(f"数据形状: (1, 3, 112, 112)")
    
    if samples_generated > 0:
        
        print(f"\n生成的文件:")
        # for file in sorted(output_dir.glob("mbf_input_*.npy")):
        #     size_kb = file.stat().st_size / 1024
        #     print(f"  - {file.name} ({size_kb:.1f} KB)")
        
        if (output_dir / "mbf_quant_all.npy").exists():
            all_file = output_dir / "mbf_quant_all.npy"
            size_mb = all_file.stat().st_size / (1024 * 1024)
            print(f"  - mbf_quant_all.npy ({size_mb:.2f} MB)")
    
    if failed_images:
        print(f"\n⚠️  失败图片: {len(failed_images)} 张")
        if len(failed_images) <= 10:
            print("失败示例:")
            for img in failed_images[:10]:
                print(f"  - {img}")
    
    print(f"{'='*60}")
    
    return output_dir

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='生成 MBF 网络量化输入数据 (使用scrfd检测器)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python %(prog)s --data-dir ./LFW --output-dir ./quant_data --num-samples 10
  
生成的文件:
  mbf_quant_all.npy                      # 合并的样本文件
        """
    )
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='包含图片的目录 (将递归搜索子目录)')
    parser.add_argument('--output-dir', type=str, default='./mbf_quant_data',
                       help='输出目录 (默认: ./mbf_quant_data)')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='需要生成的样本数量 (默认: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.isdir(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        sys.exit(1)
    
    # 生成数据
    output_dir = generate_quantization_data_with_detector(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed
    )
    

if __name__ == "__main__":
    main()