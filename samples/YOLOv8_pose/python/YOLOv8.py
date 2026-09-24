import sys
import os
import json
import time
import cv2
import numpy as np
import logging

from postprocess_numpy import PostProcess
from acuitylib.vsi_nn import VSInn
from typing import List


def init_logger():
    # init logger
    logger = logging.getLogger(__name__)    
    logger.setLevel(logging.DEBUG)  
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(' %(levelname)s %(name)s Line:%(lineno)s: %(message)s |  - %(asctime)s ')
    handler.setFormatter(formatter)
    # logger.addHandler(handler)
    if logger.hasHandlers:
        logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False
    return logger

logger = init_logger()

class YOLOv8:
    def __init__(self,
                 onnx_path,
                 batch_size=1,*,
                 conf_thresh=0.01,
                 nms_thresh=0.7):

        self.net_name = "yolov8s_pose"
        self.onnx_path = onnx_path
        self.input_shape = (1,3,640,640)

        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
            
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.agnostic = False
        self.multi_label = False
        self.max_det = 300

        self.postprocess = PostProcess(
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            agnostic=self.agnostic,
            multi_label=self.multi_label,
            max_det=self.max_det,
        )
        print(f"self.postprocess.conf_thresh: {self.postprocess.conf_thresh}")

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        self.create_nn()

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0


    def create_nn(self):
        self.nn = VSInn()
        if os.path.exists(f"./{self.net_name}.json") and os.path.exists(f"./{self.net_name}.data"):
            logger.info("Load .json & .data file")
            self.acuity_net = self.nn.create_net()
            self.nn.load_model(self.acuity_net, f"./{self.net_name}.json")
            self.nn.load_model_data(self.acuity_net, f"./{self.net_name}.data")
        
        else:
            self.acuity_net = self.nn.load_onnx(self.onnx_path,
                                            inputs="images",
                                            outputs="output0",
                                            input_size_list="3, 640, 640")
            self.nn.save_model(self.acuity_net, f"./{self.net_name}.json")
            self.nn.save_model_data(self.acuity_net, f"./{self.net_name}.data")
        
        
    def load_q_net(self, quantize_type):
        logger.info(f"quantize tyep: {quantize_type}")
        if quantize_type not in ['int8', 'uint8', 'float16', 'bfloat16', 'int16']:
            logger.error("wrong quantize type.")
            os._exit(0)
        if os.path.exists(f"./{self.net_name}_{quantize_type}.quantize") :
                    logger.info(f"Load {quantize_type} quantize file.")
                    self.nn.load_model_quantize(self.acuity_net,
                                                f"./{self.net_name}_{quantize_type}.quantize")
        else:
            logger.info(f"Quantize file not found. Please run quantize.py first.")
            raise FileNotFoundError(f"{self.net_name}_{quantize_type}.quantize does not exits.")
        
    
    def preprocess(self, ori_img):
        """
        pre-processing
        Args:
            img: numpy.ndarray -- (h,w,3)

        Returns: (3,h,w) numpy.ndarray after pre-processing

        """
        letterbox_img, ratio, (tx1, ty1) = self.letterbox(
            ori_img,
            new_shape=(self.net_h, self.net_w),
            color=(114, 114, 114),
            auto=False,
            scaleFill=False,
            scaleup=True,
            stride=32
        )

        img = letterbox_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = img.astype(np.float32)
        # input_data = np.expand_dims(input_data, 0)
        img = np.ascontiguousarray(img / 255.0)
        return img, ratio, (tx1, ty1) 

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        return im, ratio, (dw, dh)

    def predict(self, input_img, img_num):
        input_data = {self.input_name: input_img}
        outputs = self.net.process(self.graph_name, input_data)
        
        # resort
        out_keys = list(outputs.keys())
        ord = []
        for n in self.output_names:
            for i, k in enumerate(out_keys):
                if n == k:
                    ord.append(i)
                    break
        out = [outputs[out_keys[i]][:img_num] for i in ord]
        return out

    @staticmethod
    def np_to_tf(img):
        img = np.expand_dims(img, axis=0)
        import tensorflow as tf
        logger.debug(f"type of img for np_to_tf: {type(img)}")
        logger.debug(f"tuple -> np.ndarray shape: {img.shape}")
        img_tf = tf.convert_to_tensor(img)
        return img_tf
    
    def vsi_quantize_net(self, img_list:List,
                         quantize_type:str,
                         cali_batch_size:int,
                         hybrid:bool = False):
        q_er_table = {"uint8": "asymmetric_affine",
                "int8": "perchannel_symmetric_affine",
                "int16": "dynamic_fixed_point",
                "float16": "float16",
                "bfloat16": "bfloat16"}
        def get_input_for_quantize():
            for img in img_list[:cali_batch_size]:
                preprocessed_img, _, _ = self.preprocess(img)
                logger.info(f"type of img for np_to_tf: {type(preprocessed_img)}")

                single_input = []
                single_input.append(self.np_to_tf(preprocessed_img))
                
                yield single_input
                
        q_net = self.nn.quantize(self.acuity_net,
                                 qtype=quantize_type,
                                 quantizer=q_er_table[quantize_type],
                                 batch_size=1,
                                 iterations=cali_batch_size,
                                 input_generator_func=get_input_for_quantize,
                                 compute_entropy=hybrid)
        
        if hybrid:
            q_net = self.nn.quantize(q_net,
                                     qtype=quantize_type,
                                     quantizer=q_er_table[quantize_type],
                                     batch_size=1,
                                     iterations=cali_batch_size,
                                     input_generator_func=get_input_for_quantize,
                                     hybrid=True)
            
        self.nn.save_model_quantize(q_net, f"./{self.net_name}_{quantize_type}.quantize")
        
    def vsinn_infer(self, preprocessed_img_list):
        def modify_hybrid_qfile(file_path, new_content, new_file_path):
            with open(file_path, 'r', encoding='utf-8') as qfile:
                lines = qfile.readlines()
            start_index = None
            start_content = "customized_quantize_layers"
            for i, line in enumerate(lines):
                if start_content in line:
                    start_index = i
                    break
            if start_index is not None:
                lines = lines[:start_index] + [new_content]
            
            with open(new_file_path, 'w', encoding='utf-8') as new_qfile:
                new_qfile.writelines(lines)
            
        def get_input_for_infer():
            for i, preprocessed_img in enumerate(preprocessed_img_list):
                print(f">>>>>>>>>>>>>>>>>>>>>>>>>\ninfer load image:{i}\n>>>>>>>>>>>>>>>>>>>>>>>>>")
                single_input = []
                single_input.append(self.np_to_tf(preprocessed_img))
                yield single_input
        
        # 声明 VSInn 项目对象 nn， 一个nn项目可以有多个net，每个net也可以随意Load（json,data, quantize）
        
        # 用 acuity 模型推理
        # 打印输入输出端口名
        print("net inputs order:", self.nn.get_input_names(self.acuity_net))
        print("net outputs orider:", self.nn.get_output_names(self.acuity_net))
        
        outputs, batch = [], []

        for i, data in enumerate(get_input_for_infer()):
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>\nimg for infer shape: {data[0].shape}\n>>>>>>>>>>>>>>>>>>>>>>>>>>")
            ins, outs = self.nn.run_inference_session(data)
            # outs_opt = (outs[0].transpose(0,2,1),)  # opt最后的转置，可能和后处理程序对不齐。
            print(f"outs[0]:{outs[0].shape}")
            batch.append(outs[0].squeeze(0))
            # batch.append(outs[0])
        outputs.append(batch)  
        return outputs

    def __call__(self, img_list):
        self.batch_size = len(img_list)
        img_num = len(img_list)
        ori_size_list = []
        preprocessed_img_list = []
        ratio_list = []
        txy_list = []
        for ori_img in img_list:
            ori_h, ori_w = ori_img.shape[:2]
            ori_size_list.append((ori_w, ori_h))
            start_time = time.time()
            preprocessed_img, ratio, (tx1, ty1) = self.preprocess(ori_img)
            self.preprocess_time += time.time() - start_time
            preprocessed_img_list.append(preprocessed_img)
            ratio_list.append(ratio)
            txy_list.append([tx1, ty1])
        
        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(preprocessed_img_list)
        
        start_time = time.time()

        outputs = self.vsinn_infer(preprocessed_img_list)
        self.inference_time += time.time() - start_time

        start_time = time.time()
        results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
        self.postprocess_time += time.time() - start_time

        return results
