

import os

import json
import argparse
import numpy as np
from postprocess_numpy import PostProcess
from utils import COCO_CLASSES, COLORS
import logging
# logging.basicConfig(level=logging.INFO)
from acuitylib.vsi_nn import VSInn
from typing import Generator, Tuple, List, Optional
import cv2

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

class YOLO11:
    def __init__(self, args):
        self.net_name = "yolo11s"
        self.onnx_path = args.onnx_path
        
        self.input_shape = (1,3,640,640)

        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        
        # init postprocess
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
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
        
        # check batch size 
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        
        # init preprocess
        # self.use_resize_padding = True
        # self.use_vpp = False
        # self.ab = [x * self.input_scale / 255.  for x in [1, 0, 1, 0, 1, 0]]
        self.create_nn()
        


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
        
     
    def DataLoader(self, dataset_dir:str, batch_size: int) -> Generator[Tuple[List], None, None]:
        #TODO: DataLoader 应该移入yolov10 类，否则会在infer 和 quantize 两个模块暴露信息？
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
                src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)  
                if src_img is None:
                    continue
                if len(src_img.shape) != 3:
                    src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
                
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
                
            
        
    def load_q_net(self, quantize_type):
        logger.info(f"quantize tyep: {quantize_type}")
        if quantize_type not in ['int8', 'uint8', 'float16', 'bfloat16', 'int16', 'float32']:
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


    @staticmethod
    def np_to_tf(img):
        img = np.expand_dims(img, axis=0)
        import tensorflow as tf
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
        """
        预期输出是(batch_size, 84, 8400) tensor
        """
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
                # print(f">>>>>>>>>>>>>>>>>>>>>>>>>\ninfer load image:{i}\n")
                single_input = []
                single_input.append(self.np_to_tf(preprocessed_img))
                yield single_input
        
        # 声明 VSInn 项目对象 nn， 一个nn项目可以有多个net，每个net也可以随意Load（json,data, quantize）

        
        # 用 acuity 模型推理
        # 打印输入输出端口名
        # print("net inputs order:", self.nn.get_input_names(self.acuity_net))
        # print("net outputs orider:", self.nn.get_output_names(self.acuity_net))
        
        outputs, batch = [], []
        for i, data in enumerate(get_input_for_infer()):
            # print(f"img for infer shape: {data[0].shape}\n")
            ins, outs = self.nn.run_inference_session(data)
            # print(f"outs[0]:{outs[0].shape}\n>>>>>>>>>>>>>>>>>>>>>>>>>>")
            batch.append(outs[0].squeeze(0).transpose(1, 0))
        outputs.append(batch)  
        return outputs
    
    
    def postprocess_2(self, preds_batch, org_size_batch, ratios_batch, txy_batch):
        """
        Args:
            preds_bath : [[out1, out2, out3...outbatch_size]] -> dets (8400, 84)
        outputs:

            
        """
        
        def filter_boxes(outputs, conf_thresh=self.conf_thresh):
            #预测数据并转置 → [8400,84]
            # preds = outputs[0].permute(1, 0).cpu().numpy()
            # preds = outputs.transpose(1, 0)
            preds = outputs
            print(f"preds shape: {preds.shape}")
            
            # 提取置信
            boxes = preds[:, :4]        # [x_center, y_center, width, height]（相对坐标）
            obj_conf = preds[:, 4]    # 置信度
            cls_conf = preds[:, 5:]     # 类别概率
            
            # 计算最大类别得分
            max_cls_score = np.max(cls_conf, axis=1, keepdims=True)
            max_cls_idx = np.argmax(cls_conf, axis=1)
            
            # 合并置信度：obj_conf * max_cls_score


            # mask = final_scores.squeeze() > conf_thresh
            mask = max_cls_score.squeeze() > conf_thresh
            # logger.debug(f"""mask shape: {mask.shape}""")

            
            # 筛选有效框
            valid_boxes = boxes[mask]
            # logger.debug(f"""valid_boxes shape: {valid_boxes.shape}""")
            final_scores = obj_conf.reshape(8400,1) * max_cls_score
            valid_scores = final_scores[mask]
            valid_cls = max_cls_idx[mask]
            return valid_boxes, valid_scores, valid_cls        
        
        def decode_boxes(valid_boxes, scale, padding, img_shape):
            # 反归一化到输入图像尺寸（640x640）
            x_center = (valid_boxes[:, 0] - padding[0]) / scale[0]

            y_center = (valid_boxes[:, 1] - padding[1]) / scale[1]
            width = valid_boxes[:, 2] / scale[0]
            height = valid_boxes[:, 3] / scale[1]
            
            # 转换为中心坐标 → 左上角坐标
            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2
            
            # 限制边界在原始图像范围内
            xmin = np.clip(xmin, 0, img_shape[1])
            ymin = np.clip(ymin, 0, img_shape[0])
            xmax = np.clip(xmax, 0, img_shape[1])
            ymax = np.clip(ymax, 0, img_shape[0])
            return np.stack([xmin, ymin, xmax, ymax], axis=1)        
                
        def nms(boxes, scores, iou_thresh=self.nms_thresh):
            """
            Args:
            
            Outputs:
            (n,6) numpy.ndarray
            """
            
            # logger.debug(f"""
            #              boses shape: {boxes.shape}
            #              scores shape: {scores.shape}
            #              """)
            # 按分数降序排序
            # order = scores.argsort()[::-1]
            order = scores.squeeze().argsort()[::-1]
            # logger.debug(f""""
            #              scores: {scores}
            #              order: {order}""")
            boxes = boxes[order]
            scores = scores[order]
            
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                
                # 计算当前框与其他框的IoU, 以更新 order
                xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
                yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
                xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
                yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
                
                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                intersection = w * h
                
                area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
                area_j = (boxes[order[1:],2]-boxes[order[1:],0])*(boxes[order[1:],3]-boxes[order[1:],1])
                union = area_i + area_j - intersection
                
                iou = intersection / (union + 1e-6)
                inds = np.where(iou <= iou_thresh)[0]
                order = order[inds + 1]  # +1 因为order[1:]被使用了
                
                # 这是什么写法
                # keep = np.array(keep)
            return keep
            
        
        if isinstance(preds_batch, list) and len(preds_batch) == 1:
            # 1 output
            dets = np.concatenate(preds_batch)
        outs = []
        for det, (org_w, org_h), ratio, (tx1, ty1) in zip(dets, org_size_batch, ratios_batch, txy_batch):
            if det.size == 0:
                continue
            
            print(f"det type{type(det)}, det shape: {det.shape}")
            boxes, scores, cls_ids = filter_boxes(det)
            if boxes.size > 0:
                decoded_boxes = decode_boxes(boxes, ratio, (tx1, ty1), img_shape=(org_h, org_w))
                keep_indices = nms(decoded_boxes, scores)
                # 4. 最终结果
                final_boxes = decoded_boxes[keep_indices]
                final_scores = scores[keep_indices]
                final_cls = cls_ids[keep_indices]
                test = final_cls[:, None]
                logger.debug(f"""final_boxes shape: {final_boxes.shape}
                             final_scores shape: {final_scores.shape}
                             final_cls shape: {final_cls.shape}
                             test shape: {len(test)}
                             """)
                outs.append([np.hstack((final_boxes, final_scores, final_cls[:,None]))])
                logger.debug(f"""out shape: {outs[-1].shape}""")
        
        
        return np.stack(outs)




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
            preprocessed_img, ratio, (tx1, ty1) = self.preprocess(ori_img)
            # print(f"preprocessed_img shape: {preprocessed_img.shape}\tratio: {ratio}\ttxy: {tx1, ty1}")
            preprocessed_img_list.append(preprocessed_img)
            ratio_list.append(ratio)
            txy_list.append([tx1, ty1])
        
        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(preprocessed_img_list)
        

        outputs = self.vsinn_infer(preprocessed_img_list)

        results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
        # results = self.postprocess_2(outputs,ori_size_list, ratio_list, txy_list)
        

        return results


