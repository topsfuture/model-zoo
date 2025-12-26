

import os
import json
import argparse
from LPRNet import LPRNet

import logging


def save_to_json(args, res_dict, output_dir):
    # save result
    json_name = "lpr_" + args.quantize_type + "_result.json"
    with open(os.path.join(output_dir, json_name), "w") as jf:
        json.dump(res_dict, jf, indent=4, ensure_ascii=False)
    logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))

def check_dir(dir):
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except Exception as e:
            logging.error(f"fail to mkdir {dir}")
def main(args):
    check_dir(args.output_dir)
    
    lpr = LPRNet(args)
    if args.quantize_type in ['int8', 'uint8', 'float16', 'bfloat16', 'int16']:
        lpr.load_q_net(args.quantize_type)
    
    # 准备 acuity 推理模块
    lpr.nn.build_inference_session(lpr.acuity_net)
    
    results_dict = {}
    for img_list, filename_list in lpr.DataLoader(args.dataset_path, args.batch_size):
        outs = lpr(img_list)
        print(f"outs in a batch: {outs}")
        for i, filename in enumerate(filename_list):
            results_dict[filename] = outs[i]
    save_to_json(args, results_dict, args.output_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="LPRNet inference parameters")
    parser.add_argument('--onnx_path', type=str, default="../models/lprnet.onnx", help="path to onnx model")
    parser.add_argument('--inputs', type=str, default="input", help="name of input nodes")
    parser.add_argument('--outputs', type=str, default="output", help="name of outputs nodes")
    parser.add_argument('--input_size_list', type=str, default="3,24,94", help="input node shape , seperate with #")
    parser.add_argument('--dataset_path', type=str, default="../datasets/test", help="path to dataset")
    parser.add_argument('-q','--quantize_type', type=str, default="float16", choices=['int8', 'int16', 'float16', 'bfloat16', 'uint8','float'] ,help="quantize data type")
    parser.add_argument('--batch_size', type=int, default=1, help="size of batch")
    parser.add_argument('--output_dir', type=str, default="./output", help="path of output")
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    main(args)
    
