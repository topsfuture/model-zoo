#!/usr/bin/env python3

import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ACUITY_AVAILABLE = False
_VSINN_MODULE = None

def umeyama_similar_transform(src, dst, estimate_scale=True):
    num = src.shape[0]
    dim = src.shape[1]

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
        return np.float32([[1, 0, 0], [0, 1, 0]])
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

    return T[:dim, :]

class SCRFDAcuity:
    def __init__(self, model_file=None, qtype=None):
        if not ACUITY_AVAILABLE:
            raise RuntimeError("Acuity inference engine is not available.")
        self.model_file = model_file
        self.qtype = qtype
        self.taskname = 'detection'
        self.batched = False
        self.center_cache = {}
        self.nms_thresh = 0.4
        self._init_vars()
        self._init_acuity_net()

    def _init_vars(self):
        self.input_size = (640, 640)  # Model expected input size
        self.use_kps = True          # Output keypoints
        self._num_anchors = 2        # Number of anchors per position
        self.fmc = 3                 # Number of feature maps
        self._feat_stride_fpn = [8, 16, 32]  # Downsampling strides

    def _init_acuity_net(self):
        VSInn = get_vsinn()
        self.nn = VSInn()
        self.acuity_net = self.nn.create_net()

        model_dir = Path(self.model_file)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Find model files
        json_file = model_dir / "model.json"
        data_file = model_dir / "model.data"

        if not json_file.exists():
            # Try to find other possible .json files
            json_files = list(model_dir.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No .json model file found in {model_dir}")
            json_file = json_files[0]

        if not data_file.exists():
            # Try to find other possible .data files
            data_files = list(model_dir.glob("*.data"))
            if not data_files:
                raise FileNotFoundError(f"No .data model file found in {model_dir}")
            data_file = data_files[0]

        logger.info(f"Loading detection model: {json_file.name}")

        self.nn.load_model(self.acuity_net, str(json_file))
        self.nn.load_model_data(self.acuity_net, str(data_file))

        # Load quantization file based on qtype
        if self.qtype:
            quant_file_name = f"model_{self.qtype}.quantize"
            quant_file = model_dir / quant_file_name
            
            if quant_file.exists():
                self.nn.load_model_quantize(self.acuity_net, str(quant_file))
                logger.info(f"Loading quantization file: {quant_file.name}")
            else:
                logger.warning(f"Quantization file not found: {quant_file}")
                # Try to find other possible quantization files
                quantize_files = list(model_dir.glob("*.quantize"))
                if quantize_files:
                    quant_file = quantize_files[0]
                    self.nn.load_model_quantize(self.acuity_net, str(quant_file))
                    logger.info(f"Loading found quantization file: {quant_file.name}")
        else:
            logger.info("No quantization type specified, skipping quantization file loading")

        self.nn.build_inference_session(self.acuity_net)
        logger.info("Acuity detection model loading completed.")

    def prepare(self, ctx_id, **kwargs):
        nms_thresh = kwargs.get('nms_thresh')
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        input_size = kwargs.get('input_size')
        if input_size is not None:
            self.input_size = input_size
        logger.info(f"Detector ready: nms_thresh={self.nms_thresh}, input_size={self.input_size}")

    def forward(self, img, thresh):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        # Input image size (H, W)
        input_size = tuple(img.shape[0:2][::-1])

        blob = cv2.dnn.blobFromImage(img, 1.0/128, input_size,
                                    (127.5, 127.5, 127.5), swapRB=True)

        acuity_outputs = self.nn.run_inference_session([blob])
        if len(acuity_outputs) < 2:
            raise RuntimeError(f"Acuity output format abnormal, expected at least 2 elements, got {len(acuity_outputs)}")
        net_outs = acuity_outputs[1]  # Get tuple containing 9 output tensors

        # Check output count
        if len(net_outs) != 9:
            raise RuntimeError(f"Model output count abnormal, expected 9 outputs, got {len(net_outs)}")

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)

            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # Generate anchor grid
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            # Apply threshold filtering
            pos_inds = np.where(scores >= thresh)[0]

            # Decode bounding boxes
            bboxes = self._distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]

            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            if self.use_kps:
                # Decode keypoints
                kpss = self._distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list

    def _distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _distance2kps(self, points, distance, max_shape=None):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = np.clip(px, 0, max_shape[1])
                py = np.clip(py, 0, max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def detect(self, img, thresh=0.5, input_size=None, max_num=0, metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        # Maintain aspect ratio scaling and padding
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1] / input_size[0])
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]

        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, thresh)

        if not scores_list:  # No target detected
            return np.array([]), None

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale

        if self.use_kps and kpss_list:
            kpss = np.vstack(kpss_list) / det_scale
        else:
            kpss = None

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]

        if kpss is not None:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        return det, kpss

class FaceRecognizerAcuity:
    def __init__(self, model_file=None, qtype=None):
        if not ACUITY_AVAILABLE:
            raise RuntimeError("Acuity inference engine is not available.")
        self.model_file = model_file
        self.qtype = qtype
        self._init_acuity_net()

    def _init_acuity_net(self):
        VSInn = get_vsinn()
        self.nn = VSInn()
        self.acuity_net = self.nn.create_net()

        model_dir = Path(self.model_file)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        json_file = model_dir / "model.json"
        data_file = model_dir / "model.data"

        if not json_file.exists():
            json_files = list(model_dir.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No .json model file found in {model_dir}")
            json_file = json_files[0]

        if not data_file.exists():
            data_files = list(model_dir.glob("*.data"))
            if not data_files:
                raise FileNotFoundError(f"No .data model file found in {model_dir}")
            data_file = data_files[0]

        logger.info(f"Loading recognition model: {json_file.name}")

        self.nn.load_model(self.acuity_net, str(json_file))
        self.nn.load_model_data(self.acuity_net, str(data_file))

        # Load quantization file based on qtype
        if self.qtype:
            quant_file_name = f"model_{self.qtype}.quantize"
            quant_file = model_dir / quant_file_name
            
            if quant_file.exists():
                self.nn.load_model_quantize(self.acuity_net, str(quant_file))
                logger.info(f"Loading quantization file: {quant_file.name}")
            else:
                logger.warning(f"Quantization file not found: {quant_file}")
                # Try to find other possible quantization files
                quantize_files = list(model_dir.glob("*.quantize"))
                if quantize_files:
                    quant_file = quantize_files[0]
                    self.nn.load_model_quantize(self.acuity_net, str(quant_file))
                    logger.info(f"Loading found quantization file: {quant_file.name}")
        else:
            logger.info("No quantization type specified, skipping quantization file loading")

        self.nn.build_inference_session(self.acuity_net)
        logger.info("Acuity recognition model loading completed.")

    def _preprocess(self, face_img):
        face_img = cv2.resize(face_img, (112, 112))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = np.transpose(face_img, (2, 0, 1))  # HWC -> CHW
        face_img = np.expand_dims(face_img, axis=0).astype(np.float32)  # CHW -> NCHW
        face_img = (face_img - 127.5) / 128.0  # Normalization
        return face_img

    def get_embedding(self, face_img):
        input_tensor = self._preprocess(face_img)
        # Acuity inference
        acuity_outputs = self.nn.run_inference_session([input_tensor])
        # Assuming acuity_outputs[0] is the feature vector output
        if len(acuity_outputs) < 2:
            raise RuntimeError(f"Acuity output format abnormal, expected at least 2 elements, got {len(acuity_outputs)}")
        # Usually the first element is the feature vector, shape (1, 512)
        embedding = acuity_outputs[1][0]
        # L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

class BuffaloScAcuityPredictor:
    def __init__(self,
                 det_model_dir: str = "models/acuity_models/detector",
                 rec_model_dir: str = "models/acuity_models/recognizer",
                 qtype: str = None):
        logger.info(f"Initializing Acuity buffalo_sc predictor, quantization type: {qtype if qtype else 'none'}...")
        # Initialize detector
        self.detector = SCRFDAcuity(det_model_dir, qtype=qtype)
        self.detector.prepare(ctx_id=-1)  # ctx_id is meaningless for Acuity, only for interface compatibility

        # Initialize recognizer
        self.recognizer = FaceRecognizerAcuity(rec_model_dir, qtype=qtype)

        # Define standard face alignment template (ArcFace 5 points)
        self.dst_pts = np.array([
            [30.2946, 51.6963],  # Left eye
            [65.5318, 51.5014],  # Right eye
            [48.0252, 71.7366],  # Nose
            [33.5493, 92.3655],  # Left mouth corner
            [62.7299, 92.2041]   # Right mouth corner
        ], dtype=np.float32)
        logger.info("Acuity buffalo_sc predictor initialization completed.")

    def detect_faces(self, img: np.ndarray, score_threshold: float = 0.5):
        bboxes, kpss = self.detector.detect(img, thresh=score_threshold, input_size=(640, 640))

        if bboxes is None or len(bboxes) == 0:
            return [], None

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
        return processed_bboxes, kpss

    def get_embedding(self, face_img: np.ndarray):
        return self.recognizer.get_embedding(face_img)

    def get(self, img: np.ndarray, score_threshold: float = 0.3, max_faces: int = 0):
        bboxes, kpss = self.detect_faces(img, score_threshold)

        if max_faces > 0 and len(bboxes) > max_faces:
            bboxes = bboxes[:max_faces]
            if kpss is not None:
                kpss = kpss[:max_faces]

        faces = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2, score = bbox
            height, width = img.shape[:2]

            # Safe cropping boundaries
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # Face alignment
            aligned_face = None
            if kpss is not None and i < len(kpss):
                landmarks = kpss[i]
                if landmarks.shape == (5, 2):
                    try:
                        transform = umeyama_similar_transform(landmarks.astype(np.float32), self.dst_pts)
                        aligned_face = cv2.warpAffine(img, transform, (112, 112),
                                                     flags=cv2.INTER_LINEAR, borderValue=0.0)
                    except Exception as e:
                        logger.warning(f"Face alignment failed, fallback to cropping: {e}")
                        aligned_face = None

            # Fallback solution: crop and resize
            if aligned_face is None:
                face_region = img[y1:y2, x1:x2]
                if face_region.size == 0:
                    continue
                aligned_face = cv2.resize(face_region, (112, 112))

            # Extract features
            embedding = self.get_embedding(aligned_face).flatten()

            face_info = {
                'bbox': [x1, y1, x2, y2],
                'score': float(score),
                'embedding': embedding.tolist(),
                'size': embedding.shape[0]
            }
            faces.append(face_info)

        return faces

def get_vsinn():
    global ACUITY_AVAILABLE, _VSINN_MODULE
    try:
        from acuitylib.vsi_nn import VSInn
        ACUITY_AVAILABLE = True
        _VSINN_MODULE = VSInn
    except ImportError as e:
        logging.error(f"Acuity inference engine import failed: {e}")
        logging.error("Please ensure the acuitylib package is correctly installed.")
        ACUITY_AVAILABLE = False
        raise
    return _VSINN_MODULE

def create_acuity_predictor(det_model_dir: str = None,
                           rec_model_dir: str = None,
                           qtype: str = None) -> BuffaloScAcuityPredictor:
    get_vsinn()
    if det_model_dir is None:
        det_model_dir = "models/acuity_models/detector"
    if rec_model_dir is None:
        rec_model_dir = "models/acuity_models/recognizer"
    return BuffaloScAcuityPredictor(det_model_dir, rec_model_dir, qtype)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Acuity buffalo_sc model test')
    parser.add_argument('--image', type=str, default='examples/test_image.jpg',
                       help='Test image path')
    parser.add_argument('--det-model', type=str, default='models/acuity_models/detector',
                       help='Detection model directory')
    parser.add_argument('--rec-model', type=str, default='models/acuity_models/recognizer',
                       help='Recognition model directory')
    parser.add_argument('--qtype', type=str, default=None,
                       help='Quantization type, such as int8, float16, corresponding to loading model_{qtype}.quantize file')
    args = parser.parse_args()

    # Create predictor
    predictor = create_acuity_predictor(args.det_model, args.rec_model, args.qtype)

    # Read image
    img = cv2.imread(args.image)
    if img is None:
        logger.error(f"Cannot read image: {args.image}")
        exit(1)
    print(f"Image size: {img.shape}")

    # Process image
    faces = predictor.get(img, score_threshold=0.5)
    print(f"Detected {len(faces)} faces:")
    for i, face in enumerate(faces):
        bbox = face['bbox']
        score = face['score']
        emb_size = face['size']
        print(f"  Face {i + 1}:")
        print(f"    Position: ({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})")
        print(f"    Confidence: {score:.4f}")
        print(f"    Feature dimension: {emb_size}")
        
        embedding_list = face['embedding']
        if isinstance(embedding_list, list) and len(embedding_list) >= 3:
            try:
                emb_preview = embedding_list[:3]
                # Convert to float
                preview_values = [float(x) for x in emb_preview]
                print(f"    Feature preview: [{preview_values[0]:.6f}, {preview_values[1]:.6f}, {preview_values[2]:.6f}, ...]")
            except (ValueError, TypeError) as e:
                print(f"    Feature preview: Cannot format - {e}")
                print(f"    First 3 values: {emb_preview}")
        else:
            print(f"    Feature preview: Cannot display (type: {type(embedding_list)}, length: {len(embedding_list) if isinstance(embedding_list, list) else 'N/A'})")

    # Visualize results
    output_img = img.copy()
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_img, f"{face['score']:.2f}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    output_path = "result_acuity.jpg"
    cv2.imwrite(output_path, output_img)
    print(f"Result saved to: {output_path}")
