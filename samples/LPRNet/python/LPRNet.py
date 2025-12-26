

from typing import Generator, Tuple, List
import os
import cv2
import numpy as np
from acuitylib.vsi_nn import VSInn

import logging

CHARS = [
    "京",
    "沪",
    "津",
    "渝",
    "冀",
    "晋",
    "蒙",
    "辽",
    "吉",
    "黑",
    "苏",
    "浙",
    "皖",
    "闽",
    "赣",
    "鲁",
    "豫",
    "鄂",
    "湘",
    "粤",
    "桂",
    "琼",
    "川",
    "贵",
    "云",
    "藏",
    "陕",
    "甘",
    "青",
    "宁",
    "新",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "I",
    "O",
    "-",
]

CHARS_DICT = {i: char for i, char in enumerate(CHARS)}


class LPRNet(object):
    def __init__(self, args):

        self.input_shape = (1, 3, 24, 94)   # c,h,w
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        
        self.net_name = 'lprnet'
        # import params
        self.onnx_path = args.onnx_path
        self.inputs = args.inputs
        self.outputs = args.outputs
        self.input_size_list = args.input_size_list
        
        # infer params
        self.batch_size = args.batch_size
        


        self.create_nn()
        
    def create_nn(self):
        self.nn = VSInn()
        if os.path.exists(f"./{self.net_name}.json") and os.path.exists(f"./{self.net_name}.data"):
            logging.info("Load .json & .data file")
            self.acuity_net = self.nn.create_net()
            self.nn.load_model(self.acuity_net, f"./{self.net_name}.json")
            self.nn.load_model_data(self.acuity_net, f"./{self.net_name}.data")
        
        else:
            self.acuity_net = self.nn.load_onnx(self.onnx_path,
                                            inputs=self.inputs,
                                            outputs=self.outputs,
                                            input_size_list=self.input_size_list)
            self.nn.save_model(self.acuity_net, f"./{self.net_name}.json")
            self.nn.save_model_data(self.acuity_net, f"./{self.net_name}.data")

        
    def load_q_net(self, quantize_type):
        logging.info(f"quantize tyep: {quantize_type}")
        if quantize_type not in ['int8', 'uint8', 'float16', 'bfloat16', 'int16']:
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
        h, w, _ = img.shape
        if h != self.net_h or w != self.net_w:
            img = cv2.resize(img, (self.net_w, self.net_h))
        img = img.astype("float32")
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        print(f"Preprocessed image shape: {img.shape}")
        return img

    
    
    @staticmethod
    def np_to_tf(img):
        img = np.expand_dims(img, axis=0)
        import tensorflow as tf
        img_tf = tf.convert_to_tensor(img)
        return img_tf
    
    def predict(self, input_img_list):
        outputs = []
        for preprocessed_img in input_img_list:
            single_input = [self.np_to_tf(preprocessed_img)]
            ins, outs = self.nn.run_inference_session(single_input)
            outputs.append(outs[0])
        return np.concatenate(outputs, axis=0)
        

    def postprocess(self, outputs):
        res = list()
        outputs = np.argmax(outputs, axis=1)
        for output in outputs:
            print(f"Raw model output indices: {output}")
            no_repeat_blank_label = list()
            pre_c = output[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(CHARS_DICT[pre_c])
            for c in output:
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(CHARS_DICT[c])
                pre_c = c
            res.append("".join(no_repeat_blank_label))

        return res
    
    def DataLoader(self, dataset_dir:str, batch_size: int) -> Generator[Tuple[List], None, None]:
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError
        img_list = []
        filename_list = []
        results_json_list = []
        cn = 0 
        batch = 0
        for root, dirs, filenames in os.walk(dataset_dir):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg','.png','.jpeg','.bmp','.webp']:
                    continue
                img_file = os.path.join(root, filename)
                cn += 1
                # decode
                # src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)  
                src_img = cv2.imread(img_file, cv2.IMREAD_COLOR)
                # if src_img is None:
                #     continue
                # if len(src_img.shape) != 3:
                #     src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
                
                # 这里的src_img 和 filename都来自
                img_list.append(src_img)
                print(f"src_img_name:{filename}")
                filename_list.append(filename)
        
        # 2. 分batch处理
                if (len(img_list) == batch_size or cn == len(filenames)) and len(img_list):
                    yield img_list, filename_list
                    img_list.clear()
                    filename_list.clear()
                    
            if len(img_list) > 0:
                yield img_list, filename_list
                img_list.clear()
                filename_list.clear()
                

    def quantize(self, img_list:List,
                         quantize_type:str,
                         cali_batch_size:int,
                         hybrid:bool = False):
        
        quantizer_table = {"uint8": "asymmetric_affine",
                            "int8": "perchannel_symmetric_affine",
                            "int16": "dynamic_fixed_point",
                            "float16": "float16",
                            "bfloat16": "bfloat16"}
        
        def get_input_for_quantize():
            for img in img_list[:cali_batch_size]:
                preprocessed_img = self.preprocess(img)
                logging.info(f"type of img for np_to_tf: {type(preprocessed_img)}")

                single_input = []
                single_input.append(self.np_to_tf(preprocessed_img))
                
                yield single_input
                
        q_net = self.nn.quantize(self.acuity_net,
                                 qtype=quantize_type,
                                 quantizer=quantizer_table[quantize_type],
                                 batch_size=1,
                                 iterations=cali_batch_size,
                                 input_generator_func=get_input_for_quantize,
                                 compute_entropy=hybrid)
        
        if hybrid:
            q_net = self.nn.quantize(q_net,
                                     qtype=quantize_type,
                                     quantizer=quantizer_table[quantize_type],
                                     batch_size=1,
                                     iterations=cali_batch_size,
                                     input_generator_func=get_input_for_quantize,
                                     hybrid=True)
            
        self.nn.save_model_quantize(q_net, f"./{self.net_name}_{quantize_type}.quantize")
        
        
        
        
        
        
        
        
    def __call__(self, img_list):
        img_num = len(img_list)
        img_input_list = []
        for img in img_list:
            img = self.preprocess(img)
            img_input_list.append(img)

        if img_num == self.batch_size:
            input_img = np.stack(img_input_list)
            outputs = self.predict(input_img)
        else:
            input_img = np.zeros(self.input_shape, dtype="float32")
            input_img[:img_num] = np.stack(img_input_list)
            outputs = self.predict(input_img)[:img_num]

        res = self.postprocess(outputs)

        return res


