
import os
import cv2
import numpy as np
import argparse
from typing import List, Optional, Sequence, Tuple
import logging
import time
logging.basicConfig(level=logging.DEBUG)
import pyclipper
# import onnxruntime as ort
from acuitylib.vsi_nn import VSInn
from pathlib import Path

class PPOCRv2Rec(object):
    def __init__(self, args):
        # load bmodel
        
        self.net_name = "../output_rec/acuity_temp/model"
        self.nn = VSInn()
        if os.path.exists(f"./{self.net_name}.json") and os.path.exists(f"./{self.net_name}.data"):
            logging.info("Load .json & .data file")
            self.acuity_net = self.nn.create_net()
            self.nn.load_model(self.acuity_net, f"./{self.net_name}.json")
            self.nn.load_model_data(self.acuity_net, f"./{self.net_name}.data")
        
        else:
            self.acuity_net = self.nn.load_onnx(args.model_rec,
                                            inputs="x",
                                            outputs="fetch_name_0",
                                            input_size_list="3, 48, 320")
            self.nn.save_model(self.acuity_net, f"./{self.net_name}.json")
            self.nn.save_model_data(self.acuity_net, f"./{self.net_name}.data")

        quantize_file_rec = Path(args.quantize_file_rec)
        quantize_type = args.quantize_rec
        print(quantize_file_rec, quantize_type)
        if  quantize_file_rec.exists():
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
        # self.rec_batch_size = self.input_shape[0] # Max batch size in model stages.
        logging.info("load bmodel success!")
        self.img_size = args.img_size
        self.img_size = sorted(self.img_size, key=lambda x: x[0])
        self.img_ratio = [x[0]/x[1] for x in self.img_size]
        self.img_ratio = sorted(self.img_ratio)
        # 解析字符字典
        self.character = ['blank']
        with open(args.char_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character.append(line)
        if args.use_space_char:
            self.character.append(" ")
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        self.beam_search = args.use_beam_search
        self.beam_size = args.beam_size
    
    def preprocess(self, img):
        start_prep = time.time()
        h, w, _ = img.shape
        ratio = w / float(h)
        # print(f"ratio: {ratio}")
        # print(f"self.ratio: {self.img_ratio}")
        if ratio > self.img_ratio[-1]:
            logging.debug("Warning: ratio out of range: h = %d, w = %d, ratio = %f, model with larger width is recommended."%(h, w, ratio))
            resized_w = self.img_size[-1][0]
            resized_h = self.img_size[-1][1]
            padding_w = resized_w
        else:          
            for max_ratio in self.img_ratio:
                if ratio <= max_ratio:
                    resized_h = self.img_size[0][1]
                    resized_w = int(resized_h * ratio)
                    padding_w = int(resized_h * max_ratio)
                    # print("resized_w: {}, padding_w: {}, ratio: {}".format(resized_w, padding_w, ratio))
                    break
        # print(resized_w,resized_h)    
        if h != resized_h or w != resized_w:
            img = cv2.resize(img, (resized_w, resized_h))
        img = img.astype('float32')
        img = np.transpose(img, (2, 0, 1))
        img -= 127.5
        img *= 0.0078125
        
        padding_im = np.zeros((3, resized_h, padding_w), dtype=np.float32)
        
        padding_im[:, :, 0:resized_w] = img
        # print(padding_im.shape)
        self.preprocess_time += time.time() - start_prep
        return padding_im


    def predict(self, tensor):
        # inputs = self.net.get_inputs()

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

    def postprocess(self, outputs, beam_search=False, beam_width=5):
        start_post = time.time()
        result_list = []

        if beam_search:
            max_seq_len, num_classes = outputs.shape[1], outputs.shape[2]

            for batch_idx in range(outputs.shape[0]):
                beams = [{'prefix': [], 'score': 1.0, 'confs':[]}] 

                for t in range(max_seq_len):
                    new_beams = []

                    for beam in beams:

                        next_char_probs = outputs[batch_idx, t]
                        top_candidates = np.argsort(-next_char_probs)[:beam_width]

                        for c in top_candidates:
                                new_prefix = beam['prefix'] + [c]
                                new_score = beam['score'] * next_char_probs[c]
                                new_confs = beam['confs'] + [next_char_probs[c]]
                                new_beams.append({'prefix': new_prefix, 'score': new_score, 'confs':new_confs})

                    new_beams.sort(key=lambda x: -x['score'])
                    beams = new_beams[:beam_width]

                best_beam = max(beams, key=lambda x: x['score'])

                char_list = []
                conf_list = []
                pre_c = best_beam['prefix'][0]
                if pre_c != 0:
                    char_list.append(self.character[pre_c])
                    conf_list.append(best_beam['confs'][0])
                for idx, c in enumerate(best_beam['prefix']):
                    if (pre_c==c) or (c==0):
                        if c ==0:
                            pre_c = c
                        continue
                    char_list.append(self.character[c])
                    conf_list.append(best_beam['confs'][idx])
                    pre_c = c
                result_list.append((''.join(char_list), np.mean(conf_list)))

        else:  # original postprocess
            preds_idx = outputs.argmax(axis=2)
            preds_prob = outputs.max(axis=2)
            for batch_idx, pred_idx in enumerate(preds_idx):
                char_list = []
                conf_list = []
                pre_c = pred_idx[0]
                if pre_c != 0:
                    char_list.append(self.character[pre_c])
                    conf_list.append(preds_prob[batch_idx][0])
                for idx, c in enumerate(pred_idx):
                    if (pre_c == c) or (c == 0):
                        if c == 0:
                            pre_c = c
                        continue
                    char_list.append(self.character[c])
                    conf_list.append(preds_prob[batch_idx][idx])
                    pre_c = c

                result_list.append((''.join(char_list), np.mean(conf_list)))

        self.postprocess_time += time.time() - start_post
        return result_list

    def __call__(self, img_list):
        img_dict = {}
        for img_size in self.img_size:
            img_dict[img_size[0]] = {"imgs":[], "ids":[], "res":[]}
        for id, img in enumerate(img_list):
            img = self.preprocess(img)
            if img is None:
                continue
            img_dict[img.shape[2]]["imgs"].append(img)
            img_dict[img.shape[2]]["ids"].append(id)

        for size_w in img_dict.keys():
            if size_w > 640:
                for img_input in img_dict[size_w]["imgs"]:
                    img_input = np.expand_dims(img_input, axis=0)
                    outputs = self.predict(img_input)
                    res = self.postprocess(outputs,self.beam_search,self.beam_size)
                    img_dict[size_w]["res"].extend(res)
            else:
                img_num = len(img_dict[size_w]["imgs"])
                
                for i in range(img_num):
                    img_input = np.expand_dims(img_dict[size_w]["imgs"][i], axis=0)
                    outputs = self.predict(img_input)
                    res = self.postprocess(outputs,self.beam_search,self.beam_size)
                    img_dict[size_w]["res"].extend(res)   
                

        rec_res = {"res":[], "ids":[]}
        for size_w in img_dict.keys():
            rec_res["res"].extend(img_dict[size_w]["res"])
            rec_res["ids"].extend(img_dict[size_w]["ids"])
        return rec_res



def main(opt):
    ppocrv2_rec = PPOCRv2Rec(opt)
    Tp = 0
    img_list = []
    for img_name in os.listdir(opt.input):
        #print(file_name)
        #img_name = '川JK0707.jpg'
        label = img_name.split('.')[0]
        img_file = os.path.join(opt.input, img_name)
        #print(img_file, label)
        src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
        dst = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        #print(src_img.shape)
        img_list.append(dst)

    rec_res = ppocrv2_rec(img_list)

    for i, id in enumerate(rec_res.get("ids")):
        logging.info("img_name:{}, conf:{:.6f}, pred:{}".format(os.listdir(opt.input)[id], rec_res["res"][i][1], rec_res["res"][i][0]))

def img_size_type(arg):
    # 将字符串解析为列表类型
    img_sizes = arg.strip('[]').split('],[')
    img_sizes = [size.split(',') for size in img_sizes]
    img_sizes = [[int(width), int(height)] for width, height in img_sizes]
    return img_sizes

def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    
    parser.add_argument('--input', type=str, default='../datasets/cali_set_rec', help='input image directory path')
    parser.add_argument('--model_rec', type=str, default='../det_onnx/ppocr_rec.onnx', help='recognizer model path')
    parser.add_argument('--img_size', type=img_size_type, default=[[320, 48]], help='You should set inference size [width,height] manually if using multi-stage bmodel.')
    parser.add_argument("--char_dict_path", type=str, default="./ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=bool, default=True)
    parser.add_argument('--use_beam_search', action='store_const', const=True, default=False, help='Enable beam search')
    parser.add_argument("--beam_size", type=int, default=5, choices=range(1,41), help='Only valid when using beam search, valid range 1~40')
    parser.add_argument("--use-gpu", action="store_true", help="若可用则使用 CUDAExecutionProvider")
    parser.add_argument('--quantize_file_rec', type=Path, default='../output_det/acuity_temp/model_float16.quantize', help='quantize path')
    parser.add_argument('--quantize_rec', type=str, default='float16', help='quantize type')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)