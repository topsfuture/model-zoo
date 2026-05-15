
import os
import cv2
import numpy as np
from typing import List
from acuitylib.vsi_nn import VSInn
from scipy.ndimage import gaussian_filter
import logging
from math import sqrt as sqrt
from math import ceil
from itertools import product as product
import tensorflow as tf
from typing import Optional, Generator, Tuple, List


logging.basicConfig(level=logging.INFO)

cfg= {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}


class RetinaFace:
    def __init__(self, args):

        self.net_name = "retinaface"
        self.model_path = args.model_path
        self.quantize_type = args.quantize_type
        
        self.input_shape = args.input_shape
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]
        self.score_threshold = 0.5
        self.nms_threshold = 0.3
        self.mean_bgr = (104, 117, 123)
        self.min_sizes = cfg['min_sizes']

        self.steps = cfg['steps']
        self.variance = cfg['variance']
        self.clip = cfg['clip']
        self.create_acuity_net()
        
    def create_acuity_net(self):
        
        self.nn = VSInn()
        if os.path.exists(f'../model/acuity_temp/{self.net_name}.json'):
        # if 0:
            logging.info('.json ready') 
            self.acuity_net = self.nn.create_net()
            self.nn.load_model(self.acuity_net, f'../model/acuity_temp/{self.net_name}.json')
            self.nn.load_model_data(self.acuity_net, f'../model/acuity_temp/{self.net_name}.data')
        else:
            self.acuity_net = self.nn.load_onnx(self.model_path)
            self.nn.save_model(self.acuity_net, f'{self.net_name}.json')
            self.nn.save_model_data(self.acuity_net, f'{self.net_name}.data')
        
    def init_model_infer(self, args):
        if self.quantize_type != "float32":
            print(f"quantize_type is {self.quantize_type}")
            self.load_q_net(self.quantize_type)
        self.nn.build_inference_session(self.acuity_net)
        
    def load_q_net(self, quantize_type):
        logging.info(f"quantize tyep: {quantize_type}")
        if quantize_type not in ['int8', 'uint8', 'float16', 'bfloat16', 'int16']:
            logging.error("wrong quantize type.")
            raise ValueError("wrong quantize type.")
            
        if os.path.exists(f"../model/acuity_temp/{self.net_name}_{quantize_type}.quantize") :
                    logging.info(f"Load {quantize_type} quantize file.")
                    self.nn.load_model_quantize(self.acuity_net,
                                                f"./{self.net_name}_{quantize_type}.quantize")
            
        else:
            logging.info(f"Quantize file not found. Please run quantize.py first.")
            raise FileNotFoundError(f"{self.net_name}_{quantize_type}.quantize does not exits.")
        
        
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
                preprocessed_img, resize, = self.preprocess(img)
                logging.info(f"type of img for np_to_tf: {type(preprocessed_img)}")

                single_input = []
                resize_list = []
                single_input.append(self.np_to_tf(preprocessed_img))
                resize_list.append(resize)
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

    
    def DataLoader(self,dataset_path:str, batch_size:int):
        def decode_image(img_path:str):
            try:
                src_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
                if src_img is None:
                    logging(f"{img_path} imdecode is None.")
                if len(src_img.shape) != 3:  # if 4:alpha channel; if 1: gray
                    src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
                return src_img
            except Exception as e:
                logging.error(f"fail to decode image: {img_path}")
                return None
            
        # decode img
        img_list = []
        filename_list = []
        for path, dirs, filenames in os.walk(dataset_path):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg','.png','.jpeg','.bmp','.webp']:
                    continue
                src_img = decode_image(os.path.join(path, filename))
                img_list.append(src_img)
                filename_list.append(filename)
                if len(img_list) == batch_size:
                    logging.info(f"len of img_list in DataLoader(): {len(img_list)}")
                    yield img_list, filename_list
                    img_list.clear()
                    filename_list.clear()
        
            if len(img_list) > 0:    
                yield img_list, filename_list
                img_list.clear()
                filename_list.clear()
            

    
            
    def preprocess(self, ori_img):
        target_size = 640
        im_shape = ori_img.shape

        # 找出长边和短边
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # resize长变
        resize = float(target_size) / float(im_size_max)
        img = cv2.resize(ori_img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        # 补上短边
        top = 0 
        bottom =  self.net_h - img.shape[0]
        left = 0
        right = self.net_w - img.shape[1]

        pad_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        pad_img = np.float32(pad_img)

        pad_img -= self.mean_bgr
        pad_img = np.transpose(pad_img, (2, 0, 1))
        return pad_img, resize
    
    def decode(self, loc, priors, variances):
        boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes
    
    def decode_landm(self, pre, priors, variances):

        landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                            ), 1)
        return landms
    
    def prior_box(self, image_size):    
        anchors = []
        feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in self.steps]
        for k, f in enumerate(feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / image_size[1]
                    s_ky = min_size / image_size[0]
                    dense_cx = [x * self.steps[k] / image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        
        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            output = np.clip(output, 0, 1)
        
        return output
    
    def nms(self, dets, thresh):  
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
    
    def postprocess(self, outputs, img_list, resize_ratio_list):
        results = []
        confs   = []
        locs    = []
        landmss = []
        count = 0
        for output in outputs:
            for i in range(len(outputs[0])):
                if output.shape[-1] == 2:
                    confs.append(output[i])
                    confs[i] = np.expand_dims(confs[i], axis=0)
                    
                if output.shape[-1] == 4:
                    locs.append(output[i])
                    locs[i] = np.expand_dims(locs[i], axis=0)
                    
                if output.shape[-1] == 10:
                    landmss.append(output[i])
                    landmss[i] = np.expand_dims(landmss[i], axis=0)


        for i in range(len(outputs[0])):
            count +=1
            conf = confs[i]
            loc = locs[i]
            landms = landmss[i]
            resize = resize_ratio_list[i]
            scale = np.array([self.input_w, self.input_h, self.input_w, self.input_h])
            boxes = self.decode(loc.squeeze(0), self.prior_box((self.input_h, self.input_w)), cfg['variance'])
            boxes = boxes * scale / resize

            #scores = conf.squeeze(0)[:, 1]
            #提取出“目标为人脸”的置信度得分，并进行Softmax计算。
            exp_scores = np.exp(conf.squeeze(0))
            prob_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            scores = prob_scores[:, 1]

            landms = self.decode_landm(landms.squeeze(0), self.prior_box((self.input_h, self.input_w)), cfg['variance'])
            scale1 = np.array([self.input_w, self.input_h, self.input_w, self.input_h,
                           self.input_w, self.input_h, self.input_w, self.input_h,
                           self.input_w, self.input_h])
            landms = landms * scale1 / resize

            #过滤低于指定分数的预测框,inds为符合指标的索引清单
            inds = np.where(scores >= self.score_threshold)[0]
            boxes = boxes[inds]
            scores = scores[inds]
            landms = landms[inds]

            #重新排序
            order = scores.argsort()[::-1]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]
            #print(f"scores is {scores}")

            boxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = self.nms(boxes, self.nms_threshold)
            result_boxes = boxes[keep, :]
            result_landmarks = landms[keep, :]
            result_score = scores[keep]
            #print(f"keep is {keep},result_score is {result_score}")
            

            results.append((result_boxes, result_landmarks,result_score))
 
        #print(f"count is {count}")
        return results
    
    @staticmethod
    def np_to_tf(img):
        img = np.expand_dims(img, axis=0)
        logging.debug(f"type of img for np_to_tf: {type(img)}")
        logging.debug(f"tuple -> np.ndarray shape: {img.shape}")
        img_tf = tf.convert_to_tensor(img)
        return img_tf
    '''
    def vsinn_predict(self, preprocessed_img_list):
        outputs1, outputs2, outputs3 = [], [], []
        for image in preprocessed_img_list:
            data = []
            data.append(self.np_to_tf(image))
            ins, outputs = self.nn.run_inference_session(data)
            outputs1.append(outputs[0][0])
            outputs2.append(outputs[1][0])
            outputs3.append(outputs[2][0])

        return [np.array(outputs1), np.array(outputs2), np.array(outputs3)]
    '''   
    def vsinn_predict(self, preprocessed_img_list):
        outputs1, outputs2, outputs3 = [], [], []
        for image in preprocessed_img_list:
            data = []
            data.append(self.np_to_tf(image))


            ins, outputs = self.nn.run_inference_session(data)


            outputs1.append(outputs[0][0])
            outputs2.append(outputs[1][0])
            outputs3.append(outputs[2][0])
        return [np.array(outputs1), np.array(outputs2), np.array(outputs3)]

    
    def __call__(self, img_list):
        img_num = len(img_list)
        ori_size_list = []
        preprocessed_img_list = []
        resize_ratio_list = []
        for ori_img in img_list:
            ori_h, ori_w = ori_img.shape[:2]
            ori_size_list.append((ori_w, ori_h))
            preprocessed_img, resize_ratio = self.preprocess(ori_img)
            preprocessed_img_list.append(preprocessed_img)
            resize_ratio_list.append(resize_ratio)

        outs = self.vsinn_predict(preprocessed_img_list)

        results = self.postprocess(outs, img_list, resize_ratio_list)

        # 解包 results 为 boxes、landmarks、scores 列表
        boxes_list = []
        landmarks_list = []
        score_list =[] 
        for i in range(img_num):
            boxes_list.append(results[i][0])
            landmarks_list.append(results[i][1])
            score_list.append(results[i][2])

        return boxes_list, landmarks_list,score_list
