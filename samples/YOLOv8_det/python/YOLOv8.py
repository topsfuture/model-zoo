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

hybrid_layers_context2 = """customized_quantize_layers:
        model.22/Sub_output_0_11: float32
        model.22/Sub_1_output_0_7: float32
        model.0/conv/Conv_output_0_237: float32
        model.22/Add_11_output_0_8: float32
        model.22/dfl/Reshape_1_output_0_13: float32
        model.22/Div_1_output_0_6: float32
        model.22/Slice_1_output_0_15: float32
        model.22/Slice_output_0_12: float32
        model.22/Add_10_output_0_10: float32
        model.0/conv/Conv_output_0_237_acuity_mark_perm_244: float32
        
        output0_1: float32
        model.22/Concat_24_output_0_4: float32
        model.22/Mul_5_output_0_2: float32
        model.22/dfl/Softmax_output_0_18: float32
        model.22/dfl/conv/Conv_output_0_17: float32
        model.22/cv2.2/cv2.2.1/conv/Conv_output_0_49: float32
        model.22/cv2.2/cv2.2.1/act/Sigmoid_output_0_50_model.22/cv2.2/cv2.2.1/act/Mul_output_0_39: float32
        model.22/cv3.2/cv3.2.1/conv/Conv_output_0_51: float32
        model.22/cv3.2/cv3.2.1/act/Sigmoid_output_0_52_model.22/cv3.2/cv3.2.1/act/Mul_output_0_40: float32
        model.2/Concat_output_0_198: float32
        model.2/cv2/conv/Conv_output_0_191: float32
        model.2/Split_output_0_226: float32
        model.2/m.0/Add_output_0_199: float32
        model.2/m.0/cv1/conv/Conv_output_0_225: float32
        model.1/conv/Conv_output_0_233: float32
        model.1/act/Sigmoid_output_0_234_model.1/act/Mul_output_0_232: float32
        model.22/cv2.0/cv2.0.1/act/Sigmoid_output_0_42_model.22/cv2.0/cv2.0.1/act/Mul_output_0_35: float32
        model.22/cv2.0/cv2.0.2/Conv_output_0_29: float32
        model.22/cv3.0/cv3.0.1/act/Sigmoid_output_0_44_model.22/cv3.0/cv3.0.1/act/Mul_output_0_36: float32
        model.22/cv3.0/cv3.0.2/Conv_output_0_30: float32
        model.22/cv2.1/cv2.1.1/act/Sigmoid_output_0_46_model.22/cv2.1/cv2.1.1/act/Mul_output_0_37: float32
        model.22/cv2.1/cv2.1.2/Conv_output_0_31: float32
        model.22/cv3.1/cv3.1.1/act/Sigmoid_output_0_48_model.22/cv3.1/cv3.1.1/act/Mul_output_0_38: float32
        model.22/cv3.1/cv3.1.2/Conv_output_0_32: float32
        model.22/cv2.2/cv2.2.2/Conv_output_0_33: float32
        model.22/cv3.2/cv3.2.2/Conv_output_0_34: float32
        model.22/cv2.0/cv2.0.0/act/Sigmoid_output_0_54_model.22/cv2.0/cv2.0.0/act/Mul_output_0_53: float32
        model.22/cv2.0/cv2.0.1/conv/Conv_output_0_41: float32
        model.22/cv2.1/cv2.1.0/act/Sigmoid_output_0_58_model.22/cv2.1/cv2.1.0/act/Mul_output_0_57: float32
        model.22/cv2.1/cv2.1.1/conv/Conv_output_0_45: float32
        model.22/cv2.2/cv2.2.0/act/Sigmoid_output_0_62_model.22/cv2.2/cv2.2.0/act/Mul_output_0_61: float32
        model.22/cv3.2/cv3.2.0/act/Sigmoid_output_0_64_model.22/cv3.2/cv3.2.0/act/Mul_output_0_63: float32
        model.4/m.1/cv2/act/Sigmoid_output_0_144_model.4/m.1/cv2/act/Mul_output_0_143: float32
        model.4/m.1/Add_output_0_130: float32
        model.8/m.0/cv2/act/Sigmoid_output_0_177_model.8/m.0/cv2/act/Mul_output_0_176: float32
        model.8/m.0/Add_output_0_169: float32
        model.2/cv2/act/Sigmoid_output_0_192_model.2/cv2/act/Mul_output_0_186: float32
        model.3/conv/Conv_output_0_184: float32
        model.6/m.1/cv2/act/Sigmoid_output_0_203_model.6/m.1/cv2/act/Mul_output_0_195: float32
        model.6/m.1/Add_output_0_188: float32
        model.2/m.0/cv2/act/Sigmoid_output_0_214_model.2/m.0/cv2/act/Mul_output_0_206: float32
        model.2/m.0/cv1/act/Sigmoid_output_0_222_model.2/m.0/cv1/act/Mul_output_0_221: float32
        model.2/m.0/cv2/conv/Conv_output_0_213: float32
        model.2/cv1/act/Sigmoid_output_0_230_model.2/cv1/act/Mul_output_0_229: float32
        model.2/cv1/conv/Conv_output_0_231: float32
        model.0/act/Sigmoid_output_0_236_model.0/act/Mul_output_0_235: float32
        """
hybrid_layers_context = """customized_quantize_layers:
        model.22/Sub_output_0_11: dynamic_fixed_point-i16
        model.22/Sub_1_output_0_7: dynamic_fixed_point-i16
        model.0/conv/Conv_output_0_237: dynamic_fixed_point-i16
        model.22/Add_11_output_0_8: dynamic_fixed_point-i16
        model.22/dfl/Reshape_1_output_0_13: dynamic_fixed_point-i16
        model.22/Div_1_output_0_6: dynamic_fixed_point-i16
        model.22/Slice_1_output_0_15: dynamic_fixed_point-i16
        model.22/Slice_output_0_12: dynamic_fixed_point-i16
        model.22/Add_10_output_0_10: dynamic_fixed_point-i16
        model.0/conv/Conv_output_0_237_acuity_mark_perm_244: dynamic_fixed_point-i16
        model.22/Sigmoid_output_0_3: dynamic_fixed_point-i16
        model.22/dfl/Reshape_1_output_0_13_acuity_mark_perm_239: dynamic_fixed_point-i16
        model.22/dfl/Softmax_output_0_18_acuity_mark_perm_240: dynamic_fixed_point-i16
        model.22/dfl/Reshape_output_0_20: dynamic_fixed_point-i16

        attach_output0/out0_0: dynamic_fixed_point-i16
        output0_1: dynamic_fixed_point-i16
        model.22/Concat_24_output_0_4: dynamic_fixed_point-i16
        model.22/Mul_5_output_0_2: dynamic_fixed_point-i16
        model.22/dfl/Softmax_output_0_18: dynamic_fixed_point-i16
        model.22/dfl/conv/Conv_output_0_17: dynamic_fixed_point-i16
        model.22/cv2.2/cv2.2.1/conv/Conv_output_0_49: dynamic_fixed_point-i16
        model.22/cv2.2/cv2.2.1/act/Sigmoid_output_0_50_model.22/cv2.2/cv2.2.1/act/Mul_output_0_39: dynamic_fixed_point-i16
        model.22/cv3.2/cv3.2.1/conv/Conv_output_0_51: dynamic_fixed_point-i16
        model.22/cv3.2/cv3.2.1/act/Sigmoid_output_0_52_model.22/cv3.2/cv3.2.1/act/Mul_output_0_40: dynamic_fixed_point-i16
        model.2/Concat_output_0_198: dynamic_fixed_point-i16
        model.2/cv2/conv/Conv_output_0_191: dynamic_fixed_point-i16
        model.2/Split_output_0_226: dynamic_fixed_point-i16
        model.2/m.0/Add_output_0_199: dynamic_fixed_point-i16
        model.2/m.0/cv1/conv/Conv_output_0_225: dynamic_fixed_point-i16
        model.1/conv/Conv_output_0_233: dynamic_fixed_point-i16
        model.1/act/Sigmoid_output_0_234_model.1/act/Mul_output_0_232: dynamic_fixed_point-i16
        model.22/cv2.0/cv2.0.1/act/Sigmoid_output_0_42_model.22/cv2.0/cv2.0.1/act/Mul_output_0_35: dynamic_fixed_point-i16
        model.22/cv2.0/cv2.0.2/Conv_output_0_29: dynamic_fixed_point-i16
        model.22/cv3.0/cv3.0.1/act/Sigmoid_output_0_44_model.22/cv3.0/cv3.0.1/act/Mul_output_0_36: dynamic_fixed_point-i16
        model.22/cv3.0/cv3.0.2/Conv_output_0_30: dynamic_fixed_point-i16
        model.22/cv2.1/cv2.1.1/act/Sigmoid_output_0_46_model.22/cv2.1/cv2.1.1/act/Mul_output_0_37: dynamic_fixed_point-i16
        model.22/cv2.1/cv2.1.2/Conv_output_0_31: dynamic_fixed_point-i16
        model.22/cv3.1/cv3.1.1/act/Sigmoid_output_0_48_model.22/cv3.1/cv3.1.1/act/Mul_output_0_38: dynamic_fixed_point-i16
        model.22/cv3.1/cv3.1.2/Conv_output_0_32: dynamic_fixed_point-i16
        model.22/cv2.2/cv2.2.2/Conv_output_0_33: dynamic_fixed_point-i16
        model.22/cv3.2/cv3.2.2/Conv_output_0_34: dynamic_fixed_point-i16
        model.22/cv2.0/cv2.0.0/act/Sigmoid_output_0_54_model.22/cv2.0/cv2.0.0/act/Mul_output_0_53: dynamic_fixed_point-i16
        model.22/cv2.0/cv2.0.1/conv/Conv_output_0_41: dynamic_fixed_point-i16
        model.22/cv2.1/cv2.1.0/act/Sigmoid_output_0_58_model.22/cv2.1/cv2.1.0/act/Mul_output_0_57: dynamic_fixed_point-i16
        model.22/cv2.1/cv2.1.1/conv/Conv_output_0_45: dynamic_fixed_point-i16
        model.22/cv2.2/cv2.2.0/act/Sigmoid_output_0_62_model.22/cv2.2/cv2.2.0/act/Mul_output_0_61: dynamic_fixed_point-i16
        model.22/cv3.2/cv3.2.0/act/Sigmoid_output_0_64_model.22/cv3.2/cv3.2.0/act/Mul_output_0_63: dynamic_fixed_point-i16
        model.4/m.1/cv2/act/Sigmoid_output_0_144_model.4/m.1/cv2/act/Mul_output_0_143: dynamic_fixed_point-i16
        model.4/m.1/Add_output_0_130: dynamic_fixed_point-i16
        model.8/m.0/cv2/act/Sigmoid_output_0_177_model.8/m.0/cv2/act/Mul_output_0_176: dynamic_fixed_point-i16
        model.8/m.0/Add_output_0_169: dynamic_fixed_point-i16
        model.2/cv2/act/Sigmoid_output_0_192_model.2/cv2/act/Mul_output_0_186: dynamic_fixed_point-i16
        model.3/conv/Conv_output_0_184: dynamic_fixed_point-i16
        model.6/m.1/cv2/act/Sigmoid_output_0_203_model.6/m.1/cv2/act/Mul_output_0_195: dynamic_fixed_point-i16
        model.6/m.1/Add_output_0_188: dynamic_fixed_point-i16
        model.2/m.0/cv2/act/Sigmoid_output_0_214_model.2/m.0/cv2/act/Mul_output_0_206: dynamic_fixed_point-i16
        model.2/m.0/cv1/act/Sigmoid_output_0_222_model.2/m.0/cv1/act/Mul_output_0_221: dynamic_fixed_point-i16
        model.2/m.0/cv2/conv/Conv_output_0_213: dynamic_fixed_point-i16
        model.2/cv1/act/Sigmoid_output_0_230_model.2/cv1/act/Mul_output_0_229: dynamic_fixed_point-i16
        model.2/cv1/conv/Conv_output_0_231: dynamic_fixed_point-i16
        model.0/act/Sigmoid_output_0_236_model.0/act/Mul_output_0_235: dynamic_fixed_point-i16
        """



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
                 conf_thresh=0.001,
                 nms_thresh=0.7):

        self.net_name = "yolov8s"
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
