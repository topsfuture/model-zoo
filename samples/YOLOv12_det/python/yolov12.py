#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import time
import json
import argparse
import numpy as np

from postprocess_numpy import PostProcess
# import onnxruntime as ort
from acuitylib.vsi_nn import VSInn
from utils import COCO_CLASSES, COLORS
import logging
logging.basicConfig(level=logging.INFO)
import cv2
from pathlib import Path

class YOLOv12:
    def __init__(self, args):
        # load model
        self.net_name = "model"
        # self.onnx_path = args.onnx_path
        self.model_path = args.model
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
        
        # get output
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        
        # init preprocess
        # self.use_resize_padding = True
        # self.use_vpp = False
        # self.ab = [x * self.input_scale / 255.  for x in [1, 0, 1, 0, 1, 0]]
        
        # init postprocess
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        self.agnostic = False
        self.multi_label = True
        self.max_det = 300
        self.quantize_type = args.quantize_type

        self.postprocess = PostProcess(
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            agnostic=self.agnostic,
            multi_label=self.multi_label,
            max_det=self.max_det,
        )
        
        # init time
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        self.quantize = args.quantize_type
        self.quantize_file = args.quantize_file
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
                                            inputs="images",
                                            outputs="output0",
                                            input_size_list="3, 640, 640")
            self.nn.save_model(self.acuity_net, f"./{self.net_name}.json")
            self.nn.save_model_data(self.acuity_net, f"./{self.net_name}.data")

        quantize_file = Path(self.quantize_file)
        quantize_type = self.quantize
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
        # so = ort.SessionOptions()
        # # so.intra_op_num_threads = args.intra_threads
        # # so.inter_op_num_threads = args.inter_threads
        # so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        # available = set(ort.get_available_providers())
        # providers: List[str] = []
        # provider_options: List[dict] = []

        # providers.append("CPUExecutionProvider")
        # provider_options.append({})

        # try:
        #     session = ort.InferenceSession(
        #         str(self.model_path),
        #         so,
        #         providers=providers,
        #         provider_options=provider_options,
        #     )
        # except Exception as exc:  # pragma: no cover - 保护性日志
        #     raise RuntimeError(f"初始化 ONNXRuntime 会话失败: {exc}") from exc
        
        # return session

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
    def predict(self, input_tensor, img_num = 1):
        """
        ensure output order: loc_data, conf_preds, mask_data, proto_data
        Args:
            input_tensor:
        Returns:
        """
        
        # inputs = self.net.get_inputs()
        # if not inputs:
        #     raise ValueError("ONNX 模型未包含输入节点")
        # input_name = inputs[0].name

        # ins,outputs = self.net.run_inference_session(input_tensor)
        
        def get_input_for_infer():
            for i, preprocessed_img in enumerate(input_tensor):
                # print(f">>>>>>>>>>>>>>>>>>>>>>>>>\ninfer load image:{i}\n")
                single_input = []
                single_input.append(self.np_to_tf(preprocessed_img))
                yield single_input
        
        # 声明 VSInn 项目对象 nn， 一个nn项目可以有多个net，每个net也可以随意Load（json,data, quantize）

        scale = 2.5896198749542236  # 常见量化scale
        zero_point = -121         # 常见量化零点
        # 用 acuity 模型推理
        # 打印输入输出端口名
        # print("net inputs order:", self.nn.get_input_names(self.acuity_net))
        # print("net outputs orider:", self.nn.get_output_names(self.acuity_net))
        # print("net inputs order:", input_tensor.shape)
        outputs, batch = [], []
        for i, data in enumerate(get_input_for_infer()):
            # print(f"img for infer shape: {data[0].shape}\n")
            ins, outs = self.nn.run_inference_session(data)
            print(f"outs[0]:{outs[0].shape}\n>>>>>>>>>>>>>>>>>>>>>>>>>>")
            
            # batch.append(outs[0])
            output = outs[0].squeeze(0)
            # print(f"----------------------------")
            # for i in range(output.shape[0]):
            #     print(f"output[{i}]: {output[i]}")
            # print(f"----------------------------")
            if self.quantize_type == 'int8':
                float32_output = (output - zero_point) * scale
                # print(f"=================================")
                # for i in range(float32_output.shape[0]):
                #     print(f"float32_output[{i}]: {float32_output[i]}")
                # print(f"=================================")
                float32_output = float32_output.astype(np.float32)
            elif self.quantize_type == 'float16':
                # float32_output = output.astype(np.float32)
                float32_output = output
            
            # for i in range(float32_output.shape[0]):
            #     print(f"float32_output[{i}]: {float32_output[i]}")
            
            batch.append(float32_output)
        outputs = np.array(batch)
        # outputs = np.array(batch)
        print(f"outputs: {outputs[0].shape}")  
        # print(f"outputs: {outputs[0]}")
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
        

        # outputs = self.vsinn_infer(preprocessed_img_list)
        outputs = self.predict(preprocessed_img_list)
        results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
        # results = self.postprocess_2(outputs,ori_size_list, ratio_list, txy_list)

        return results
        
def draw_cv(image, boxes, output_img_dir, file_name, cn, masks=None, classes_ids=None, conf_scores=None, isvideo=False):


    thickness = 2
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        if classes_ids is not None:
            print(int(classes_ids[idx]) + 1)
            color = np.array(COLORS[int(classes_ids[idx]) + 1]).astype(np.uint8).tolist()
        else:
            color = (0, 0, 255)
        if (x2 - x1) <= thickness * 2 or (y2 - y1) <= thickness * 2:
            logging.info("width or height too small, this rect will not be drawed: (x1={},y1={},w={},h={})".format(x1, y1, x2-x1, y2-y1))
        else:
            cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness)
        cv2.putText(image, COCO_CLASSES[int(classes_ids[idx] + 1)], (x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1.0, tuple(color),1)
        logging.debug("class id={}, score={}, (x1={},y1={},w={},h={})".format(int(classes_ids[idx]), conf_scores[idx], x1, y1, x2-x1, y2-y1))

        cv2.imwrite(os.path.join(output_img_dir, file_name), image)

def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.model):
        raise FileNotFoundError('{} is not existed.'.format(args.model))
    
    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir) 

    yolov12 = YOLOv12(args)
    batch_size = 1
    

    decode_time = 0.0
    # test images
    if os.path.isdir(args.input): 
        img_list = []
        filename_list = []
        results_list = []
        cn = 0
        
        for root, dirs, filenames in os.walk(args.input):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg','.png','.jpeg','.bmp','.webp']:
                    continue
                img_file = os.path.join(root, filename)
                cn += 1
                logging.info("{}, img_file: {}".format(cn, img_file))
                # decode
                start_time = time.time()
                src_img = cv2.imread(img_file)
             
                decode_time += time.time() - start_time
                img_list.append(src_img)
                filename_list.append(filename)
                # if (len(img_list) == batch_size or cn == len(filenames)) and len(img_list):
                if len(img_list) :
                    # predict
                    results = yolov12(img_list)
                    for i, filename in enumerate(filename_list):
                        det = results[i]
                        # save image
                        det_draw = det[det[:, -2] > 0.1]
                        draw_cv(
                                  img_list[i],
                                  det_draw[:,:4],
                                  output_img_dir,
                                  filename,
                                  cn, 
                                  masks=None,
                                  classes_ids=det_draw[:, -1],
                                  conf_scores=det_draw[:, -2])
                        print("save image done")
                        # save result
                        res_dict = dict()
                        res_dict['image_name'] = filename
                        res_dict['bboxes'] = []
                        for idx in range(det.shape[0]):
                            bbox_dict = dict()
                            x1, y1, x2, y2, score, category_id = det[idx]
                            bbox_dict['bbox'] = [float(round(x1, 3)), float(round(y1, 3)), float(round(x2 - x1,3)), float(round(y2 -y1, 3))]
                            bbox_dict['category_id'] = int(category_id)
                            bbox_dict['score'] = float(round(score,5))
                            res_dict['bboxes'].append(bbox_dict)
                        results_list.append(res_dict)
                        
                    img_list.clear()
                    filename_list.clear()

        # save results
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        json_name = os.path.split(args.model)[-1] + "_" + os.path.split(args.input)[-1] + "_cv" + "_python_result.json"
        with open(os.path.join(output_dir, json_name), 'w') as jf:
            # json.dump(results_list, jf)
            json.dump(results_list, jf, indent=4, ensure_ascii=False)
        logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))
    
    # test videos
    else:
       
        logging.info("result saved in {}".format(output_img_dir))


    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = yolov12.preprocess_time / cn
    inference_time = yolov12.inference_time / cn
    postprocess_time = yolov12.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--model', type=str, default='', help='path of model')
    parser.add_argument('--quantize_file', type=str, default="", help='quantize file path')
    parser.add_argument('--quantize_type', type=str, default="int8", help="quantize data type", choices=['int8', 'uint8', 'float16', 'bfloat16', 'int16'])
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='nms threshold')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')



