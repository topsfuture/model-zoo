#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import cv2
import argparse

import math
import logging
import time
logging.basicConfig(level=logging.DEBUG)


from acuitylib.vsi_nn import VSInn
from shapely.geometry import Polygon
import pyclipper
# import onnxruntime as ort
import numpy as np


import json

from pathlib import Path
import sys



class DBPostProcess(object):
    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.6,
                 max_candidates=1000,
                 unclip_ratio=1.5,
                 use_dilation=False,
                 score_mode="fast",
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        assert score_mode in [
            "slow", "fast"
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]
        num_contours = min(len(contours), self.max_candidates)
        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue
            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.float32))
            scores.append(score)
        return np.array(boxes, dtype=np.float32), scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        '''
        box_score_fast: use bbox mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps']
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        # print(pred.shape)
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                   src_w, src_h)
            boxes_batch.append({'points': boxes})
        
        print(f"检测到 {len(boxes)} 个文本框")
        for idx, box in enumerate(boxes):
            flattened = ", ".join(f"{p:.2f}" for p in box.reshape(-1))
            print(f"  Box {idx}: [{flattened}]")
        return boxes_batch   
    
        
class PPOCRv2Det(object):
    def __init__(self, args):
        # load bmodel
        
        model_path = Path(args.model_det)
        # if not model_path.exists():
        #     raise FileNotFoundError(f"未找到 ONNX 模型文件: {model_path}")
        self.net_name = "../output_det/acuity_temp/model"
        self.nn = VSInn()
        if os.path.exists(f"./{self.net_name}.json") and os.path.exists(f"./{self.net_name}.data"):
            logging.info("Load f{self.net_name}.json & .data file")
            self.acuity_net = self.nn.create_net()
            self.nn.load_model(self.acuity_net, f"./{self.net_name}.json")
            self.nn.load_model_data(self.acuity_net, f"./{self.net_name}.data")
        
        else:
            self.acuity_net = self.nn.load_onnx(self.onnx_path,
                                            inputs="x",
                                            outputs="sigmoid_0.tmp_0",
                                            input_size_list="3, 480, 480")
            self.nn.save_model(self.acuity_net, f"./{self.net_name}.json")
            self.nn.save_model_data(self.acuity_net, f"./{self.net_name}.data")

        quantize_file = Path(args.quantize_file_det)
        quantize_type = args.quantize_det
        print(quantize_file, quantize_type)
        if  quantize_file.exists():
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
        self.nn.build_inference_session(self.acuity_net)
        # self.input_shape = self.net.get_max_input_shapes(self.graph_name)[self.input_name]
        # preprocess
        self.det_limit_side_len = args.limit_side_len
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)).astype('float32') * 255.0
        self.scale = np.array([1/0.229, 1/0.224, 1/0.225]).reshape((1, 1, 3)).astype('float32') * 1 / 255.0
        self.count = 0
        # postprocess
        self.postprocess_op = DBPostProcess(thresh=0.3,
                                            box_thresh=0.6,
                                            max_candidates=1000,
                                            unclip_ratio=1.5,
                                            use_dilation=False,
                                            score_mode="fast")
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        self.disable_normalize = args.disable_normalize
    
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
    def preprocess(self, img):
        h, w= img.shape[:2]
        size_max = max(h, w)
        if size_max >= self.det_limit_side_len:
            limit_side_len = self.det_limit_side_len
            ratio = float(limit_side_len) / size_max
        else:
            for side_len in self.det_limit_side_len:
                if size_max <= side_len:
                    limit_side_len = side_len
                    ratio = 1.
                    break
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
        # print(resize_h, resize_w)
        # print(h,w)
        # resize_h = 480
        # resize_w = 480
        if h != resize_h or w != resize_w:
            img = cv2.resize(img, (resize_w, resize_h))
        # cv2.imwrite("input.jpg",img)
        img = img.astype('float32')
        img = img - self.mean
        img = img * self.scale
        # if self.disable_normalize:
        #     norm = img
        # else:
        #     norm = img / 255.0
        #     norm = (norm - self.mean) * self.scale
        img = np.transpose(img, (2, 0, 1))

        padding_im = np.zeros((3, limit_side_len, limit_side_len), dtype=np.float32)
        padding_im[:, 0 : resize_h, 0 : resize_w] = img

        return padding_im.astype(np.float32), [h, w, resize_h, resize_w]
    
    def predict(self, tensor):
        # inputs = self.net.get_inputs()
        # if not inputs:
        #     raise ValueError("ONNX 模型未包含输入节点")
        # input_name = inputs[0].name
        # print(tensor.shape)
        ins ,outputs = self.nn.run_inference_session([tensor])
        # print(outputs[0].shape)
        if not outputs:
            raise ValueError("ONNX 模型未返回输出")
        prob = outputs[0]
        # prob = prob.squeeze()
        # if prob.ndim != 2:
        #     raise ValueError(f"期望得到 2D 概率图，实际维度 {prob.shape}")
        return prob.astype(np.float32)
    
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
        
    def postprocess(self, outputs, src_h, src_w, resize_h, resize_w):
        preds = {}
        # preds['maps'] = np.expand_dims(outputs[ 0,0,resize_h, resize_w], axis=0)
        feature_map = outputs[:, 0, 0:resize_h, 0:resize_w]  # 形状：(N, resize_h, resize_w)

        # 增加批次维度（若N=1，此时feature_map已含批次维度，可省略expand_dims）
        preds['maps'] = np.expand_dims(feature_map, axis=0)  # 最终形状：(1, N, resize_h, resize_w)
        shape_list = np.array([src_h, src_w, resize_h / float(src_h), resize_w / float(src_w)])
        shape_list = np.expand_dims(shape_list, axis=0)
        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        
        dt_boxes = self.filter_tag_det_res(dt_boxes, (src_h, src_w, 3))
        # print(dt_boxes)
        return dt_boxes
    
    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        
        return dt_boxes
    
    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points
    
    
    def __call__(self, img_list):
        img_num = len(img_list)
        img_input_list = []
        img_size_list = []
        # 对每张图片进行预处理
        start_prep = time.time()
        for img in img_list:
            img, [src_h, src_w, resize_h, resize_w] = self.preprocess(img)
            
            img_input_list.append(img)
            img_size_list.append([src_h, src_w, resize_h, resize_w])
        self.preprocess_time += time.time() - start_prep
        outputs_list = []
        start_infer = time.time()
        for img_src in img_input_list:
            img_input = np.expand_dims(img_src, axis=0)
            outputs = self.predict(img_input)
            
            outputs_list.append(outputs)
        # np.save("bin/det_preprocessed_{}".format(self.count), img_input)
         
        self.inference_time += time.time() - start_infer
        # 对输出进行后处理
        start_post = time.time()
        dt_boxes_list = []
        
        for i in range(0,img_num):
            
            src_h, src_w, resize_h, resize_w = img_size_list[i]
            # print(img_size_list[i])
            dt_boxes = self.postprocess(outputs_list[i], src_h, src_w, resize_h, resize_w)
            dt_boxes_list.append(dt_boxes)
        self.postprocess_time += time.time() - start_post
        return dt_boxes_list

def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        print(box)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im

def _collect_images(image_path: Path | None, image_dir: Path | None) -> List[Path]:
    image_paths: List[Path] = []
    if image_dir is not None:
        if not image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {image_dir}")
        if not image_dir.is_dir():
            raise ValueError(f"--image-dir 必须指向目录: {image_dir}")
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
        for pattern in patterns:
            image_paths.extend(sorted(image_dir.glob(pattern)))
    if image_path is not None:
        if not image_path.exists():
            raise FileNotFoundError(f"图像不存在: {image_path}")
        if image_path.is_dir():
            raise ValueError("--image 参数不能是目录，请改用 --image-dir")
        image_paths.append(image_path)

    if not image_paths:
        raise ValueError("请至少通过 --image 或 --image-dir 指定一张图像")
    return image_paths

def main(opt):
    draw_img_save = "./results/det_results"
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    ppocrv2_det = PPOCRv2Det(opt)
    # 读取得到的图片存放在这个list中
    file_list = _collect_images(opt.image, opt.image_dir)
    img_list = []
    for img_name in file_list:
        
        src_img = cv2.imread(str(img_name),cv2.IMREAD_COLOR)
        img_list.append(src_img)
    # 检测得到的结果
    dt_boxes_list = ppocrv2_det(img_list)
    print(dt_boxes_list)
    for img_name, dt_boxes in zip(file_list, dt_boxes_list):
        print(img_name)
        print(dt_boxes)
        
        draw_im = draw_text_det_res(dt_boxes, str(img_name))
        img_name_pure = os.path.split(img_name)[-1]
        img_path = os.path.join(draw_img_save,
                                "det_res_{}".format(img_name_pure))
        cv2.imwrite(img_path, draw_im)
        logging.info("The visualized image saved in {}".format(img_path))
def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    
    parser.add_argument('--image_dir', type=Path,  help='input image directory path')
    parser.add_argument('--image', type=Path, help='input image ')
    parser.add_argument('--model_det', type=Path, default='../ppocr_det.onnx', help='model path')
    parser.add_argument('--quantize_file', type=Path, default='../output_det/acuity_temp/model_int8.quantize', help='quantize path')
    parser.add_argument('--quantize', type=str, default='int8', help='quantize type')
    
    parser.add_argument("--keep-bgr", action="store_true", help="保持 BGR 输入（默认转换为 RGB）")
    parser.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406], help="归一化均值")
    parser.add_argument("--std", nargs=3, type=float, default=[0.229, 0.224, 0.225], help="归一化标准差")
    parser.add_argument("--disable-normalize", action="store_true", help="关闭 /255 与均值方差归一化")
    parser.add_argument("--limit-side-len", type=int, default=480, help="最长（或最短）边 resize 上限")
    parser.add_argument("--limit-type", choices=["max", "min"], default="max", help="基于最长或最短边进行缩放")

    parser.add_argument("--bin-thresh", type=float, default=0.3, help="DB 二值化阈值")
    parser.add_argument("--box-thresh", type=float, default=0.6, help="候选框得分阈值")
    parser.add_argument("--max-candidates", type=int, default=1000, help="最大候选框数量")
    parser.add_argument("--unclip-ratio", type=float, default=1.5, help="外扩系数")
    parser.add_argument("--min-size", type=float, default=3.0, help="最短边过滤阈值")
    parser.add_argument("--use-dilation", action="store_true", help="二值化后执行形态学膨胀")
    parser.add_argument("--dilation-kernel", type=int, default=2, help="膨胀核尺寸")

    parser.add_argument("--use-gpu", action="store_true", help="若可用则使用 CUDAExecutionProvider")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA 设备号（仅在可用时生效）")
    parser.add_argument("--intra-threads", type=int, default=4, help="ONNXRuntime intra-op 线程数")
    parser.add_argument("--inter-threads", type=int, default=1, help="ONNXRuntime inter-op 线程数")
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)