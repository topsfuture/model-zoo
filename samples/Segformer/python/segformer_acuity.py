import custom as dp  # 自定义的图像处理模块
# import onnxruntime as ort
from acuitylib.vsi_nn import VSInn
from typing import Generator, Tuple, List, Optional
import logging
logging.basicConfig(level=logging.INFO)

import json
import logging
import argparse
from pathlib import Path
import os

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)


class Segformer:
    """Segformer模型推理器"""
    
    def __init__(self, model_path: str):
        self.model_name = 'segformer'
        self.input_shape = (1, 3, 512, 1024)
        self.mean = [123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
        # self.size_divisor = 32
        
        # 创建ORT推理会话
        # providers = ['CPUExecutionProvider']
        # self.session = ort.InferenceSession(model_path, providers=providers)
        # self.input_name = self.session.get_inputs()[0].name
        # self.output_names = [output.name for output in self.session.get_outputs()]
        
        # logging.debug(f"模型 {model_path} 加载成功")
        
        # 创建 acuity 推理会话
        self.create_nn()
        self.load_q_net('float16')
        self.nn.build_inference_session(self.acuity_net)
        
    def create_nn(self):
        self.nn = VSInn()
        self.net_name = 'model'
        if os.path.exists(f"./{self.net_name}.json") and os.path.exists(f"./{self.net_name}.data"):
            logging.info("Load .json & .data file")
            self.acuity_net = self.nn.create_net()
            self.nn.load_model(self.acuity_net, f"./{self.net_name}.json")
            self.nn.load_model_data(self.acuity_net, f"./{self.net_name}.data")
    
    def load_q_net(self, quantize_type):
        logging.info(f"quantize tyep: {quantize_type}")
        if quantize_type not in ['int8', 'uint8', 'float16', 'bfloat16', 'int16', 'float32']:
            logging.error("wrong quantize type.")
            os._exit(0)
        if os.path.exists(f"./{self.net_name}_{quantize_type}.quantize") :
                    logging.info(f"Load {quantize_type} quantize file.")
                    self.nn.load_model_quantize(self.acuity_net,
                                                f"./{self.net_name}_{quantize_type}.quantize")
        else:
            logging.info(f"Quantize file not found. Please run quantize.py first.")
            raise FileNotFoundError(f"{self.net_name}_{quantize_type}.quantize does not exits.")
        
        
        
        

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        self.ori_size = image.shape
        
        # 转换为RGB并调整大小
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_image, (self.input_shape[3], self.input_shape[2]))
        
        # 归一化处理
        normalized = (resized.astype(np.float32) - self.mean) / self.std
        
        # 转换为NCHW格式
        nchw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(nchw, axis=0).astype(np.float32)

    def postprocess(self, original_img: np.ndarray, output: np.ndarray) -> np.ndarray:
        """后处理：生成分割结果"""
        logging.debug(f"输出形状: {output.shape}")
        
        # 调整原始图像大小
        resized_ori = cv2.resize(original_img, (self.input_shape[3], self.input_shape[2]))
        
        # 应用调色板
        result = dp.palette_img(resized_ori, output[0])
        
        # 恢复原始尺寸
        if self.ori_size != result.shape:
            h, w = self.ori_size[0], self.ori_size[1]
            result = cv2.resize(result, (w, h))
            
        return result

    def __call__(self, src_img: np.ndarray) -> tuple:
        """推理单张图像"""
        # ort inference
        # input_tensor = self.preprocess(src_img)
        # output = self.session.run(self.output_names, {self.input_name: input_tensor})[0]
        # segmentation = self.postprocess(src_img, output)

        # acuity inference
        input_tensor = self.preprocess(src_img)
        output = self.nn.run_inference_session([input_tensor])[1][0]
        segmentation = self.postprocess(src_img, output)

        return output, segmentation

def get_image_paths(image_dir: str) -> list:
    """获取目录下所有图片路径"""
    image_paths = []
    image_extensions = ('.jpg', '.jpeg', '.png')
    
    for root, _, files in os.walk(image_dir):
        logging.info(f"搜索目录: {root}")
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
                
    return image_paths


def process_single_image(segformer: Segformer, image_path: str, output_dir: str, 
                        data_info: dict) -> dict:
    """处理单张图像并保存结果"""
    filename = os.path.basename(image_path)
    src_img = cv2.imread(image_path)
    
    if src_img is None or len(src_img.shape) != 3 or src_img.shape[2] != 3:
        logging.error(f"图片读取失败或格式错误: {image_path}")
        return None
    
    # 推理
    output, seg = segformer(src_img)
    filename_no_ext = os.path.splitext(filename)[0]
    
    # 保存原始输出
    output_file = os.path.join(output_dir, 'result_cl', f'{filename_no_ext}.png')
    output_2d = output[0, 0].astype(np.uint8)
    cv2.imwrite(output_file, output_2d)
    
    # 保存可视化结果
    seg_file = os.path.join(output_dir, 'images', filename)
    cv2.imwrite(seg_file, seg)
    
    # 构建结果字典
    relative_path = os.path.relpath(image_path, data_info['img_dir'])
    seg_map = relative_path.replace(data_info['img_suffix'], data_info['seg_map_suffix'])
    
    return {
        "res": output_file,
        "filename": relative_path,
        "ann": {"seg_map": seg_map}
    }


def process_cityscapes_dataset(segformer: Segformer, args: argparse.Namespace) -> dict:
    """处理整个Cityscapes数据集"""
    # 准备输出目录
    output_dir = './result/'
    os.makedirs(os.path.join(output_dir, 'result_cl'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # 数据集配置
    dataset_config = {
        'data_root': 'cityscapes',
        'img_dir': 'leftImg8bit/val',
        'ann_dir': 'gtFine/val',
        'img_suffix': '_leftImg8bit.png',
        'seg_map_suffix': '_gtFine_labelTrainIds.png'
    }
    
    # 构建完整路径
    data_root = os.path.join(args.input, dataset_config['data_root'])
    img_dir = os.path.join(data_root, dataset_config['img_dir'])
    ann_dir = os.path.join(data_root, dataset_config['ann_dir'])
    
    logging.info(f"图片目录: {img_dir}")
    
    # 获取所有图片路径
    image_paths = get_image_paths(img_dir)
    logging.info(f"共发现 {len(image_paths)} 张图片")
    
    # 处理每张图片
    results = []
    dataset_config['img_dir'] = img_dir
    dataset_config['ann_dir'] = ann_dir
    
    for i, image_path in enumerate(image_paths, 1):
        logging.info(f"处理进度: {i}/{len(image_paths)}")
        
        result = process_single_image(segformer, image_path, output_dir, dataset_config)
        if result:
            results.append(result)
    
    return {
        "data_root": data_root,
        "img_dir": dataset_config['img_dir'],
        "ann_dir": dataset_config['ann_dir'],
        "img_info": results
    }


def save_results(results: dict, args: argparse.Namespace):
    output_dir = './result/'
    model_name = os.path.basename(args.model).replace('.onnx', '')
    input_name = os.path.basename(args.input.rstrip('/'))
    
    json_name = f"{model_name}_{input_name}_opencv_python_result.json"
    json_path = os.path.join(output_dir, json_name)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    logging.info(f"结果已保存至: {json_path}")


def main(args: argparse.Namespace):
    # 初始化模型
    segformer = Segformer(args.model)
    
    # 处理数据集
    results = process_cityscapes_dataset(segformer, args)
    
    # 保存结果
    save_results(results, args)


def parse_args():
    parser = argparse.ArgumentParser(description='Segformer模型推理')
    parser.add_argument('--input', type=str, default='datasets/',
                        help='输入图片目录路径')
    parser.add_argument('--model', type=str, default='../segformer_sh_sim.onnx',
                        help='ONNX模型路径')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)