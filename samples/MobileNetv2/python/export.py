

import argparse
from mobilenetv2 import MOBILENETv2
from typing import Tuple

def parse_args():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument("--onnx_path", type=str, default="../mobilenetv2.bin", help="path of model")
    parser.add_argument("--dataset_path", type=str, default="../datasets/imagenet_val_1k/img", help="path of dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
    parser.add_argument("--input_shape", type=Tuple[int, int, int, int], default=(1, 3, 224, 224), help="input shape of model")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="Confidence threshold for postprocess")
    parser.add_argument("--nms_thresh", type=float, default=0.25, help="NMS threshold for postprocess")
    parser.add_argument("--quantize_type", type=str, default="float16", choices=['int8','uint8', 'int16', 'float16','float32'], help="Quantize type")
    args = parser.parse_args()
    return args

def main(args):
    mobilenetv2 = MOBILENETv2(args.onnx_path, args.input_shape, args.conf_thresh, args.nms_thresh, args.quantize_type)
    
    if args.quantize_type != 'float32':
        mobilenetv2.load_q_model(args.quantize_type)
        
    mobilenetv2.nn.export_ovxlib(mobilenetv2.acuity_net,
                            output_path = './export/',
                            pack_nbg_unify = True,
                            optimize = 'VIP9200O_PID0X10000049',
                            viv_sdk = '/workspace/Unified_Driver/cmdtools')
    
    


if __name__ == '__main__':
    args = parse_args()
    main(args)