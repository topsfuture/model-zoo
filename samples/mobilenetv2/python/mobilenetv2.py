import os

import json
import argparse
import numpy as np
# from postprocess_numpy import PostProcess
import logging
# logging.basicConfig(level=logging.INFO)
from acuitylib.vsi_nn import VSInn
from typing import Generator, Tuple, List, Optional
import cv2

logging.basicConfig(level=logging.INFO)

class MOBILENETv2:
    def __init__(self, model_path, input_shape, conf_thresh, nms_thresh, quantize_type, net_name:str = "mobilenetv2"):
        self.net_name = net_name
        self.model_path = model_path
        self.input_shape = input_shape
        logging.info(f"input_shape: {self.input_shape}")
        
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.quantize_type = quantize_type
        
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        
        # need to change mean and std for mobilenetv2
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        # self.mean = [0.5, 0.5, 0.5]
        # self.std = [0.5, 0.5, 0.5]

        self.create_nn()

    def create_nn(self):
        self.nn = VSInn()
        if os.path.exists(f"./{self.net_name}.json"):
            logging.info(".json file already exists.Load.")
            self.acuity_net = self.nn.create_net()
            self.nn.load_model(self.acuity_net, f"./{self.net_name}.json")
            self.nn.load_model_data(self.acuity_net, f"./{self.net_name}.data")
        else:
            # need to change function because mobilenetv2 use pytorch model 
            self.acuity_net = self.nn.load_onnx(self.model_path,
                                                 inputs="input",
                                                 outputs="output",
                                                 input_size_list = "3,224,224")
            self.nn.save_model(self.acuity_net, f"./{self.net_name}.json")
            self.nn.save_model_data(self.acuity_net, f"./{self.net_name}.data")
    
    def load_model(self):
        logging.info(f"quantize_type: {self.quantize_type}")
        if self.quantize_type in ["int8", "uint8", "int16"]:
            if os.path.exists(f"./{self.net_name}_{self.quantize_type}.quantize"):
                logging.info(f"{self.quantize_type} quantize file already exists.Load.")
                self.nn.load_model_quantize(self.acuity_net, f"./{self.net_name}_{self.quantize_type}.quantize")
            else:
                logging.info(f"Quantize file does not exist. Please run quantize.py first.")
                raise FileNotFoundError(f"{self.net_name}_{self.quantize_type}.quantize does not exist.")
        
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
        
    
    @staticmethod
    def np_to_tf(img):
        """load的时候是np array, vsi_nn推理基于tf"""
        img = np.expand_dims(img, axis=0)
        import tensorflow as tf
        img_tf = tf.convert_to_tensor(img, dtype=tf.float32)
        return img_tf

    def vsi_nn_quantize_net(self, img_list, quantize_type:str, net_name:str, onnx_path:str, cali_batch_size:int, hybrid:bool = False):
        q_er_table = {"int8": "asymmetric_affine",
                      "uint8": "asymmetric_affine",
                      "int16": "dynamic_fixed_point"}
        def get_input_for_quantize():
            for i, img in enumerate(img_list[:cali_batch_size]):
                preprocessed_img = self.preprocess(img)
                single_input = []
                single_input.append(self.np_to_tf(preprocessed_img))
                yield single_input
        
        q_mobilenetv2 = self.nn.quantize(self.acuity_net,
                                        qtype=quantize_type,
                                        quantizer=q_er_table[quantize_type],
                                        batch_size = 1,
                                        iterations=cali_batch_size,
                                        input_generator_func=get_input_for_quantize,
                                        compute_entropy=True)
        
        if hybrid:
            q_mobilenetv2 = self.nn.quantize(q_mobilenetv2,
                                            qtype=quantize_type,
                                            quantizer=q_er_table[quantize_type],
                                            batch_size = 1,
                                            iterations=cali_batch_size,
                                            input_generator_func=get_input_for_quantize,
                                            hybrid=True,
                                            )
        self.nn.save_model_quantize(q_mobilenetv2, f"./mobilenetv2_{quantize_type}.quantize")

    def preprocess(self, img):
        h, w, _ = img.shape
        if h != self.net_h or w != self.net_w:
            img = cv2.resize(img, (self.net_w, self.net_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        img = (img/255-self.mean)/self.std
        img = np.transpose(img, (2, 0, 1))
        return img
    
    def vsi_nn_infer(self, img_input_list: list, path_to_quantize_file:str = None):
        def get_input_for_infer():
            for i, preprocessed_img in enumerate(img_input_list):
                single_input = []
                single_input.append(self.np_to_tf(preprocessed_img))
                yield single_input
                
        outputs, batch_output = [], []
        # self.nn.build_inference_session(self.acuity_net)
        for i, data in enumerate(get_input_for_infer()):
            logging.info(f"img_for infer shape: {data[0].shape}\n------------------")
            ins, outs = self.nn.run_inference_session(data)
            # return outs[0]
            logging.info(f"outs[0]: {outs[0].shape}")
            batch_output.append(outs[0])
        return np.concatenate(batch_output)
        
        outputs.extend([np.concatenate(batch_output)])
        return outputs
    

    # def postprocess(self, outputs):
    #     logging.info(f"outputs shape: {outputs.shape}, outputs.dtype: {outputs.dtype}")
    #     res = list()
    #     # outputs_exp = np.exp(outputs)
    #     # outputs = outputs_exp / np.sum(outputs_exp, axis=1)[:,None]
    #     predictions = np.argmax(outputs, axis = 1)[0]
    #     logging.info(f"predictions: {predictions}, outputs shape: {outputs.shape}")
    #     res.append((predictions.tolist(), float(outputs[0][predictions])))
    #     # for pred, output in zip(predictions, outputs):
    #     #     score = output[pred]
    #     #     # logging.info(f"pred shape: {pred.shape}, score shape: {score.shape}")
    #     #     res.append((pred.tolist(),float(score)))
    #     # logging.info(f"res shape: {len(res)}")
    #     logging.info(f"res: {res}")
    #     return res 

    def postprocess(self, outputs):
        logging.info(f"outputs shape: {outputs.shape}, outputs.dtype: {outputs.dtype}")
        res = []

        output_scale = 0.11949687451124191   
        output_zero_point = -53    

        outputs = outputs.astype(np.float32)
        dequant_outputs = (outputs - output_zero_point) * output_scale

        exp_outputs = np.exp(dequant_outputs - np.max(dequant_outputs, axis=1, keepdims=True))
        softmax_outputs = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)

        predictions = np.argmax(softmax_outputs, axis=1)[0]
        confidence = float(softmax_outputs[0][predictions])

        res.append((int(predictions), confidence))
        logging.info(f"predictions: {predictions}, confidence: {confidence:.6f}")
        return res

    def infer_image(self, img_list, path_to_quantize_file = None):
        img_num = len(img_list)
        img_input_list = []
        for img in img_list:
            img = self.preprocess(img)
            img_input_list.append(img)
        
        # if img_num == self.batch_size:
        #     input_img = np.stack(img_input_list)
        #     outputs = self.predict(input_img)
        # else:
        #     input_img = np.zeros(self.input_shape, dtype='float32')
        #     input_img[:img_num] = np.stack(img_input_list)
        #     outputs = self.predict(input_img)[:img_num]
         
        # DataLoader 负责batch 分流
        logging.info(f"img_input_list shape: {len(img_input_list)}")
        outputs = self.vsi_nn_infer(img_input_list, path_to_quantize_file)
        logging.info(f"outputs shape: {outputs.shape}")
        res = self.postprocess(outputs)

        return res
    
    def get_time(self):
        return self.dt