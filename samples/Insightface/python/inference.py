#!/usr/bin/env python3
"""
buffalo_sc 模型推理接口
支持单张图片人脸检测和特征提取
"""

import os
import cv2
import numpy as np
from urllib.request import urlretrieve
import zipfile
import onnxruntime
from scrfd import SCRFD

def umeyama_similar_transform_cpp(src, dst, estimate_scale=True):
    num = src.shape[0]
    dim = src.shape[1]
    
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    
    A = np.dot(dst_demean.T, src_demean) / float(num)
    
    d = np.ones((dim, 1), dtype=np.float32)
    if np.linalg.det(A) < 0:
        d[dim - 1, 0] = -1
    
    T = np.eye(dim + 1, dtype=np.float32)
    
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    V = Vt.T  # 获取V（Vt的转置），以匹配OpenCV的V
    
    # 计算矩阵的秩
    rank = np.linalg.matrix_rank(A)
    
    if rank == 0:
        T[:dim, :dim] = np.eye(dim, dtype=np.float32)
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:  
            T[:dim, :dim] = np.dot(U, V)  # 第一次赋值
        else:
            s = d[dim - 1, 0] = -1
            d[dim - 1, 0] = -1  
            
            T[:dim, :dim] = np.dot(U, V) 
            
            diag_ = np.diag(d.flatten())
            twp = np.dot(diag_, V)  
            
            B = np.zeros((3, 3), dtype=np.uint8)
            C = np.diag(B)  
            
            T[:dim, :dim] = np.dot(U, twp)  # U * twp
            
            d[dim - 1, 0] = s  
    else:
        diag_ = np.diag(d.flatten())
        twp = np.dot(diag_, Vt)  
        res = np.dot(U, twp)  
        
        # 最终赋值
        T[:dim, :dim] = np.dot(np.dot(U, diag_), V)
    
    # 计算方差
    var_ = np.var(src_demean, axis=0)
    val = np.sum(var_)
    
    res = d.flatten() * S
    
    # 缩放计算
    if estimate_scale:
        scale = 1.0 / val * np.sum(res)
    else:
        scale = 1.0
    
    temp1 = np.dot(T[:dim, :dim], src_mean.T)
    temp2 = scale * temp1
    temp3 = dst_mean.T - temp2
    
    for i in range(min(dim, 2)):
        T[i, dim] = temp3[i, 0]
    
    # 应用缩放
    T[:dim, :dim] *= scale
    
    # 返回2x3矩阵
    return T[:dim, :]

def umeyama_similar_transform(src, dst, estimate_scale=True):
    num = src.shape[0]
    dim = src.shape[1]

    # 中心化
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = np.dot(dst_demean.T, src_demean) / num
    d = np.ones((dim,), dtype=np.float32)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float32)
    U, S, Vt = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)

    if rank == 0:
        return np.float32([[1, 0, 0], [0, 1, 0]])  # 返回单位矩阵
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            T[:dim, :dim] = np.dot(U, Vt)
        else:
            s = d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), Vt))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), Vt))

    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(src_mean, T[:dim, :dim].T)
    T[:dim, :dim] *= scale

    return T[:dim, :]  # 返回 2x3 仿射变换矩阵

class BuffaloScPredictor:
    """buffalo_sc 模型预测器"""
    # 模型下载地址
    MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/buffalo_sc/buffalo_sc.zip"
    
    def __init__(self, model_dir="../models/", device="cpu"):
        """
        初始化预测器
        
        Args:
            model_dir: 模型存储目录
            device: 推理设备，'cpu' 或 'cuda'
        """
        self.model_dir = model_dir
        self.device = device
        self.detector = None
        self.recognizer = None
        
        # 确保模型存在
        self._ensure_models()
        
        # 初始化模型
        self._init_models()
    
    def _ensure_models(self):
        """确保模型文件存在，如不存在则自动下载"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 检查是否已有模型文件
        onnx_files = [f for f in os.listdir(self.model_dir) if f.endswith('.onnx')]
        if len(onnx_files) >= 2:  # buffalo_sc 应包含至少2个onnx文件
            print(f"模型文件已存在于: {self.model_dir}")
            return
        
        # 下载模型
        print(f"下载 buffalo_sc 模型到 {self.model_dir}...")
        zip_path = os.path.join(self.model_dir, "buffalo_sc.zip")
        
        try:
            urlretrieve(self.MODEL_URL, zip_path)
            print("下载完成，正在解压...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.model_dir)
            
            os.remove(zip_path)
            print("模型准备就绪")
        except Exception as e:
            print(f"模型下载失败: {e}")
            print("请手动从以下地址下载模型并解压到 models/ 目录:")
            print(self.MODEL_URL)
            raise
    def _init_models(self):
        """初始化ONNX模型"""
        # 设置ONNX Runtime执行提供者
        providers = ['CPUExecutionProvider']
        if self.device.lower() == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # 查找模型文件
        model_files = {}
        for f in os.listdir(self.model_dir):
            if f.endswith('.onnx'):
                if 'det' in f.lower() or 'scrfd' in f.lower():
                    model_files['detector'] = os.path.join(self.model_dir, f)
                else:
                    model_files['recognizer'] = os.path.join(self.model_dir, f)
        
        if len(model_files) < 2:
            raise FileNotFoundError(f"在 {self.model_dir} 中未找到足够的模型文件")
        
        # 初始化检测器
        print(f"加载检测模型: {os.path.basename(model_files['detector'])}")
        self.detector = SCRFD(model_file=model_files['detector'])
        
        # 准备检测器，设置推理设备
        ctx_id = -1 if self.device.lower() == 'cpu' else 0
        self.detector.prepare(ctx_id=ctx_id)
        
        # 初始化识别器
        print(f"加载识别模型: {os.path.basename(model_files['recognizer'])}")
        self.recognizer = onnxruntime.InferenceSession(
            model_files['recognizer'], 
            providers=providers
        )
        
        # 获取识别器的输入名称
        self.rec_input_name = self.recognizer.get_inputs()[0].name
        
        print(f"模型初始化完成，使用设备: {self.device}")
    
    def _preprocess_detection(self, img, target_size=(640, 640)):
        """预处理图片用于人脸检测"""
        # 计算缩放比例
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(target_size[1]) / target_size[0]
        
        if im_ratio > model_ratio:
            new_height = target_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = target_size[0]
            new_height = int(new_width * im_ratio)
        
        # 调整大小
        resized_img = cv2.resize(img, (new_width, new_height))
        
        # 填充到目标尺寸
        det_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        
        # 转换为RGB，归一化，调整维度顺序
        det_img = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
        det_img = np.transpose(det_img, (2, 0, 1))  # HWC -> CHW
        det_img = np.expand_dims(det_img, axis=0).astype(np.float32)  # CHW -> NCHW
        det_img = (det_img - 127.5) / 128.0  # 归一化
        
        return det_img, (new_width, new_height)
    
    def _preprocess_recognition(self, face_img):
        """预处理人脸图片用于识别"""
        # 调整到112x112
        face_img = cv2.resize(face_img, (112, 112))
        
        # 转换为RGB，归一化，调整维度顺序
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = np.transpose(face_img, (2, 0, 1))  # HWC -> CHW
        face_img = np.expand_dims(face_img, axis=0).astype(np.float32)  # CHW -> NCHW
        face_img = (face_img - 127.5) / 128.0  # 归一化
        
        return face_img
    
    def detect_faces(self, img, score_threshold=0.5):
        """
        检测图片中的人脸
        
        Args:
            img: 输入图片 (BGR格式)
            score_threshold: 置信度阈值
            
        Returns:
            List of bounding boxes: [[x1, y1, x2, y2, score], ...]
        """
        if self.detector is None:
            raise RuntimeError("检测器未初始化")
        
        # 预处理
        bboxes, kpss = self.detector.detect(img, thresh=score_threshold, input_size=(640, 640))
        
        if bboxes is None or len(bboxes) == 0:
            return [], None
        
        # 将边界框坐标转换为整数
        processed_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2, score = bbox
            processed_bboxes.append([
                int(round(x1)), 
                int(round(y1)), 
                int(round(x2)), 
                int(round(y2)), 
                float(score)
            ])
        
        # 始终返回两个值
        return processed_bboxes, kpss
        
    
    def get_embedding(self, face_img):
        """
        提取单张人脸图片的特征向量
        
        Args:
            face_img: 已裁剪的人脸图片 (BGR格式)

        Returns:
            归一化后的512维特征向量
        """
        if self.recognizer is None:
            raise RuntimeError("识别器未初始化")
        
        # 预处理
        processed_img = self._preprocess_recognition(face_img)
        
        # 执行推理
        outputs = self.recognizer.run(None, {self.rec_input_name: processed_img})
        
        # 获取特征向量
        embedding = outputs[0][0]  # 假设第一个输出是特征向量
        
        # 归一化特征向量
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def get(self, img, score_threshold=0.3, max_faces=0):
        """
        综合接口: 检测人脸并提取特征
        
        Args:
            img: 输入图片 (BGR格式)
            score_threshold: 人脸检测置信度阈值
            max_faces: 最大处理人脸数，0表示不限制
            
        Returns:
            List of face information dicts
        """
        # 检测人脸
        bboxes, kpss = self.detect_faces(img, score_threshold)
        
        # 限制人脸数量
        if max_faces > 0 and len(bboxes) > max_faces:
            bboxes = bboxes[:max_faces]
            if kpss is not None:
                kpss = kpss[:max_faces]
        
        faces = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2, score = bbox
            
            # 将浮点数坐标转换为整数
            x1_int = int(round(x1))
            y1_int = int(round(y1))
            x2_int = int(round(x2))
            y2_int = int(round(y2))
            
            # 确保坐标在图片范围内
            height, width = img.shape[:2]
            x1_int = max(0, min(x1_int, width-1))
            y1_int = max(0, min(y1_int, height-1))
            x2_int = max(0, min(x2_int, width-1))
            y2_int = max(0, min(y2_int, height-1))
            
            # 确保x2 > x1 且 y2 > y1
            if x2_int <= x1_int or y2_int <= y1_int:
                continue
            
            # 人脸对齐（如果有关键点）
            aligned_face = None
            if kpss is not None and i < len(kpss):
                # 获取当前人脸的5个关键点
                landmarks = kpss[i]  # 形状应为 (5, 2)
                
                # 定义标准人脸模板 (ArcFace 使用的5点对齐模板，在112x112图像上的坐标)
                # 顺序通常为: 左眼, 右眼, 鼻子, 左嘴角, 右嘴角
                dst_pts = np.array([
                    [30.2946, 51.6963],  # 左眼
                    [65.5318, 51.5014],  # 右眼
                    [48.0252, 71.7366],  # 鼻子
                    [33.5493, 92.3655],  # 左嘴角
                    [62.7299, 92.2041]   # 右嘴角
                ], dtype=np.float32)
                
                # 确保我们有正确的5个点
                if landmarks.shape == (5, 2):
                    src_pts = landmarks.astype(np.float32)
                    
                    # 计算仿射变换矩阵 (将检测到的点映射到模板点)
                    # 使用 estimateAffinePartial2D (相似变换，保持角度) 或 getAffineTransform
                    try:
                        transform = umeyama_similar_transform(src_pts, dst_pts)
                        # transform, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                        # transform = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
                        # 应用变换，得到 112x112 的对齐后人脸
                        aligned_face = cv2.warpAffine(img, transform, (112, 112), flags=cv2.INTER_LINEAR, borderValue=0.0)
                    except Exception as e:
                        print(f"人脸对齐失败，将使用裁剪方案: {e}")
                        aligned_face = None
            
            # 如果对齐失败或没有关键点，则回退到裁剪+缩放
            if aligned_face is None:
                face_region = img[y1_int:y2_int, x1_int:x2_int]
                if face_region.size == 0:
                    continue
                aligned_face = cv2.resize(face_region, (112, 112))
            # --- 对齐部分结束 ---
            
            # 提取特征
            embedding = self.get_embedding(aligned_face)  # get_embedding 内部会进行预处理
            
            # 构建人脸信息
            face_info = {
                'bbox': [x1_int, y1_int, x2_int, y2_int],
                'score': float(score),
                'embedding': embedding.tolist(),
                'size': embedding.shape[0]
            }
            
            faces.append(face_info)
        
        return faces
    
def create_predictor(device='cpu'):
    """创建预测器的便捷函数"""
    return BuffaloScPredictor(device=device)

if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 使用默认测试图片
        image_path = "examples/test_image.jpg"
        if not os.path.exists(image_path):
            print(f"请提供图片路径，或创建测试图片: {image_path}")
            sys.exit(1)
    
    # 创建预测器
    predictor = create_predictor(device='cpu')
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        sys.exit(1)
    
    print(f"图片尺寸: {img.shape}")
    
    # 处理图片
    faces = predictor.get(img, score_threshold=0.5)
    
    print(f"\n检测到 {len(faces)} 张人脸:")
    for i, face in enumerate(faces):
        bbox = face['bbox']
        score = face['score']
        emb_size = face['size']
        
        print(f"  人脸 {i+1}:")
        print(f"    位置: ({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})")
        print(f"    置信度: {score:.4f}")
        print(f"    特征维度: {emb_size}")
        
        # 显示前3个特征值
        emb_preview = face['embedding'][:3]
        print(f"    特征预览: [{emb_preview[0]:.6f}, {emb_preview[1]:.6f}, {emb_preview[2]:.6f}, ...]")
    
    # 可视化结果
    output_img = img.copy()
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_img, f"{face['score']:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 保存结果
    output_path = "result.jpg"
    cv2.imwrite(output_path, output_img)
    print(f"\n结果已保存到: {output_path}")