
import os
from YOLO11 import YOLO11
from argparse import ArgumentParser




def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--onnx_path', type=str, default="../yolo11.onnx", help="path to onnx model")
    parser.add_argument('--dataset_path', type=str, default="../val2017_1000", help="path to dataset")
    parser.add_argument('--input_shape', type=tuple, default=(1, 3, 640, 640), help="input shape of acuity model")
    parser.add_argument('--quantize_type', type=str, default="float16", help="quantize data type", choices=['int8', 'uint8', 'float16', 'bfloat16', 'int16'])
    args = parser.parse_args()
    
    return args



def main(args):
    yolo11 = YOLO11(args)
    if args.quantize_type not in  ['int8', 'uint8', 'float16', 'bfloat16', 'int16']:
        print()
        os.exit(0)
    
    yolo11.load_q_net(args.quantize_type)
    
    yolo11.nn.export_ovxlib(yolo11.acuity_net,
                            output_path = './export/',
                            pack_nbg_unify = True,
                            optimize = 'VIP9200O_PID0X10000049',
                            viv_sdk = '/workspace/Unified_Driver/cmdtools')

if __name__ == "__main__":
    args = parse_args()
    main(args)
