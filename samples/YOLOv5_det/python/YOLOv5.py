
import os
import json
import time
import cv2
import argparse
import numpy as np
from utils import COLORS, COCO_CLASSES, ANCHORS, multiclass_nms, make_grid, _sigmoid
from typing import List
from acuitylib.vsi_nn import VSInn

import logging
logging.basicConfig(level=logging.INFO)

class YOLOv5:
    def __init__(self, args):

        self.net_name = "yolov5"
        self.onnx_path = args.onnx_path
        self.quantize_type = args.quantize_type
        
        self.input_shape = args.input_shape
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        

        self.agnostic = False
        self.multi_label = True
        self.max_det = 1000
        self.create_acuity_net()
        
        
    def create_acuity_net(self):
        
        self.nn = VSInn()
        if os.path.exists(f'./{self.net_name}.json'):
            logging.info('.json ready')
            self.acuity_net = self.nn.create_net()
            self.nn.load_model(self.acuity_net, f'./{self.net_name}.json')
            self.nn.load_model_data(self.acuity_net, f'./{self.net_name}.data')
        else:
            self.acuity_net = self.nn.load_onnx(self.onnx_path,
                inputs='images',
                outputs='output 350 498 646',
                input_size_list='3,640,640')
            self.nn.save_model(self.acuity_net, f'{self.net_name}.json')
            self.nn.save_model_data(self.acuity_net, f'{self.net_name}.data')
        
    def init_model_infer(self, args):
        if self.quantize_type != "float32":
            self.load_q_net(self.quantize_type)
        
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        self.nn.build_inference_session(self.acuity_net)
        
    def load_q_net(self, quantize_type):
        logging.info(f"quantize tyep: {quantize_type}")
        if quantize_type not in ['int8', 'uint8', 'float16', 'bfloat16', 'int16']:
            logging.error("wrong quantize type.")
            raise ValueError("wrong quantize type.")
            
        if os.path.exists(f"./{self.net_name}_{quantize_type}.quantize") :
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
                preprocessed_img, _, _ = self.preprocess(img)
                logging.info(f"type of img for np_to_tf: {type(preprocessed_img)}")

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
    
    @staticmethod
    def DataLoader(dataset_path:str, batch_size:int):
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

    
    @staticmethod
    def postproc(outputs, input_shape, top, left,ratio, anchors=ANCHORS):
        z = []
        # for out in outputs.values():                                        # 字典取value
        for out in outputs:                                                  # 列表取值
            if out.ndim != 5 or (out.shape[0], out.shape[1], out.shape[4]) != (1, 3, 85):
                if out.ndim == 4 and (out.shape[0], out.shape[1]) == (1, 255):
                    out = out.reshape(1, 3, 85, out.shape[2], out.shape[3])
                    out = out.transpose(0, 1, 3, 4, 2)
                elif out.ndim == 4 and (out.shape[0], out.shape[3]) == (1, 255):
                    out = out.reshape(1, out.shape[1], out.shape[2], 3, 85)
                    out = out.transpose(0, 3, 1, 2, 4)
                elif out.ndim == 4 and (out.shape[0], out.shape[1], out.shape[3] % 85) == (1, 3, 0):
                    out = out.reshape(1, 3, out.shape[2], -1, 85)
                else:
                    # print("Warning: Output node with shape {} is not vaild, please check.".format(out.shape))
                    continue
            # （1, 3, 80/40/20，80/40/20， 85）
            _, _, ny, nx, _ = out.shape  # x,y 顺序?  确实是yx
            # print(out.shape)
            # 640/80 640/40 640/20
            stride = input_shape[0] / ny   # 8/16/32
            assert (stride == input_shape[1] / nx)
            anchor = anchors[stride]
            grid, anchor_grid = make_grid(ny, nx, stride, anchor)

            y = _sigmoid(out)                    


            '''
            for resize-pad
            '''
            #yolo输出的box，0,2 - y-w
            # grid -> pixel , xywh
            y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + grid) * stride      # xy 框左上角相对当前grid左上角的位置公式。y[..., 0/1] yx 乘上stride后，单位是pixel, grid的故事就变成了pixel的故事。
            # print(left, top)

            # y[..., 0] = y[..., 0] - left   # w    left, top不是0吗。而且-left和top 与 rw/rh的stretch 复原冲突的。现在没有预处理Pad，直接原图输入Inferen的
            # y[..., 1] = y[..., 1] - top  # h  做pad了吗？没做就不用-pad.top

            
            y[..., 2:4] = (y[..., 2:4] * 2)**2 * anchor_grid  # wh 框的wh

            # y[..., 0:4:2] /= rw   # y 确认。0,2是w. 不同工具在内存中存的格式不同 
            # y[..., 1:4:2] /= rh   # h 。在这里resize可能加大误差。
            z.append(y.reshape(-1, 85))   #z.shape=(19200,85) 抽象上是什么排序。80*80*3，什么顺序排成行？下标（门牌）顺序
        pred = np.concatenate(z, axis=0)   # 将z按照下标0的方向上拼接成85列长列，什么顺序？ boxes[: [0,2]] 是x值，但是对应的是rw，这是为何？
        boxes = pred[:, :4] # 0-3  pres.shape= (25200, 85)
        conf = pred[:, 4]

        # boxes /= ratio

        # print("ratio is :%f left is : %f top is : %f\n" %(ratio, left, top))
        # print("ratio is :%f left is : %d top is : %d\n" , %(ratio, left, top))
        # print('boxed is: \n',boxes)
        scores = pred[:, 4:5] * pred[:, 5:] # Pc * 80类 ，3(anchor) * 6400(grid)  [25200,80]
        
        # xywh -> xyxy
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.

        # rh = float(input_shape[0] / imsize[0])
        # rw = float(input_shape[1] / imsize[1])
        # boxes_xyxy[:, [0,2]] /= rw
        # boxes_xyxy[:, [1,3]] /= rh
        """
        输入是output字典，{(1200, 85), (4800, 85), (19200, 85)}
        输出是scores列表，shape(25200, 80); boxes_xyxy的bounding box,shape (25200, 4)
        """
        
        logging.info(f"boxes_xyxy shape: {boxes_xyxy.shape}, conf shape: {conf.shape}, scores shape: {scores.shape}")
        # result = boxes_xyxy + [conf] + scores
        return boxes_xyxy, conf, scores
 
 
    
    
    def postprocess(self, outs_list, txy_list, ratio_list):
        """
        outs_list: outs in a batch
        dets: np.ndarray, shape=(n, 6), dim1:[x1, y1, x2, y2, cls_score, cls_index]
        """
        result_list = []
        for i, out in enumerate(outs_list):
            
            # result = self.postproc(out, (self.net_h, self.net_w), txy_list[i][0], txy_list[i][1], ratio_list[i])
            # print(result)
            boxes_xyxy, conf, class_scores = self.postproc(out, (self.net_h, self.net_w), txy_list[i][0], txy_list[i][1], ratio_list[i])
            # print(f"{result[5:]}")
            dets = multiclass_nms(boxes_xyxy,
                                class_scores,
                                iou_thres=self.nms_thresh,
                                conf_thres=self.conf_thresh,
                                class_agnostic=True)
            
            result_list.append(dets)   # list of ndarray, element:[x1,y1,w,h,scores, category_ids]
        return result_list
    
    @staticmethod
    def np_to_tf(img):
        img = np.expand_dims(img, axis=0)
        import tensorflow as tf
        logging.debug(f"type of img for np_to_tf: {type(img)}")
        logging.debug(f"tuple -> np.ndarray shape: {img.shape}")
        img_tf = tf.convert_to_tensor(img)
        return img_tf
    
    def vsinn_predict(self, preprocessed_img_list, img_num):
        def get_input_for_infer():
            for i, preprocessed_img in enumerate(preprocessed_img_list):
                # print(f">>>>>>>>>>>>>>>>>>>>>>>>>\ninfer load image:{i}\n>>>>>>>>>>>>>>>>>>>>>>>>>")
                single_input = []
                single_input.append(self.np_to_tf(preprocessed_img))
                yield single_input
    
        print("net inputs order:", self.nn.get_input_names(self.acuity_net))
        print("net outputs orider:", self.nn.get_output_names(self.acuity_net))
        
        outputs, batch = [], []     # batch in inference, depend on param: input_size. not costomized
        
        # self.nn.build_inference_session(self.acuity_net)    # 会有build 打印，是否应该
        
        for i, data in enumerate(get_input_for_infer()):
            # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>\nimg for infer shape: {data[0].shape}\n>>>>>>>>>>>>>>>>>>>>>>>>>>")
            ins, outs = self.nn.run_inference_session(data)
            # outs_opt = (outs[0].transpose(0,2,1),)  # opt最后的转置，可能和后处理程序对不齐。
            # print(f"shape of out port 1: {outs[1].shape}")
            # os._exit(0)
            batch = outs[1:]   # 丢弃 out0 的输出
            outputs.append(batch)  
        return outputs


    
    def __call__(self, img_list):
        img_num = len(img_list)
        ori_size_list = []
        preprocessed_img_list = []
        ratio_list = []
        txy_list = []
        for ori_img in img_list:
            ori_h, ori_w = ori_img.shape[:2]
            ori_size_list.append((ori_w, ori_h))
            preprocessed_img, ratio, (tx1, ty1) = self.preprocess(ori_img)
            preprocessed_img_list.append(preprocessed_img)
            ratio_list.append(ratio)
            txy_list.append([tx1, ty1])
        
        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(preprocessed_img_list)
        logging.info(f"input_img shape: {input_img.shape}\nimg num: {img_num}")
        outs = self.vsinn_predict(input_img, img_num)    
        logging.info(f"outs in __call__() len:{len(outs)}")
        
        # yolo out :[x1,y1,w,h, conf, each_class_scores] for each grid
        # 先做nms, conf 滤除，再做坐标的浮点计算，grid-xyxy, 然后再做逆变换resize
        result_list = self.postprocess(outs, txy_list, ratio_list)      
        # result: np.ndarray shape (n,6), dim1 变量顺序： [x1, y1, x2, y2, class_scores, class_ids]
        
        results = []
        for i in range(len(result_list)):
            logging.info(f"result_list in __call__() len: {len(result_list)} ")
            if result_list[i] is not None:
                results.append(np.array(result_list[i]))
            else:
                results.append(np.empty((0,6)))
        for det, (org_w, org_h), ratio, (tx1, ty1) in zip(results, ori_size_list, ratio_list, txy_list):
            if len(det):
                # Rescale boxes from img_size to im0 size
                coords = det[:, :4]
                coords[:, [0, 2]] -= tx1  # x padding
                coords[:, [1, 3]] -= ty1  # y padding
                coords[:, [0, 2]] /= ratio[0]
                coords[:, [1, 3]] /= ratio[1]

                coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, org_w - 1)  # x1, x2
                coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, org_h - 1)  # y1, y2

                det[:, :4] = coords.round()
        return results      # list of ndarray, element:[x1,y1,w,h,class_ids,scores]
