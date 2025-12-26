
import os
import argparse
from LPRNet import LPRNet


def main(args):
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError
    lpr = LPRNet(args)
    
    dataloader = lpr.DataLoader(args.dataset_path, args.batch_size)
    img_list, _ = next(dataloader)
    print(f"img_list length: {len(img_list)}")
    print(f"img_list[0] shape: {img_list[0].shape}")

    lpr.quantize(img_list, 
                args.quantize_type,
                len(img_list),
                args.hybrid)




def parse_args():
    parser = argparse.ArgumentParser(description="LPRNet inference parameters")
    parser.add_argument('--onnx_path', type=str, default="../lprnet.onnx", help="path to onnx model")
    parser.add_argument('--inputs', type=str, default="input.1", help="name of input nodes")
    parser.add_argument('--outputs', type=str, default="135", help="name of outputs nodes")
    parser.add_argument('--input_size_list', type=str, default="3,24,94", help="input node shape , seperate with #")
    parser.add_argument('--dataset_path', type=str, default="../datasets/test", help="path to dataset")
    parser.add_argument('-q','--quantize_type', type=str, default="float16", choices=['int8', 'int16', 'float16', 'bfloat16', 'uint8','float'] ,help="quantize data type")
    parser.add_argument('--batch_size', type=int, default=5, help="size of batch")
    parser.add_argument('--output_dir', type=str, default="./output", help="path of output")
    parser.add_argument('--hybrid', type=bool, default=False, help="hybrid quantize or not ")
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    main(args)
    
