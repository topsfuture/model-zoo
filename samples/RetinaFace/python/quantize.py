import os
import argparse
#from samples.OpenPose.python.OpenPose import  RetinaFace
from RetinaFace import RetinaFace
import logging
logging.basicConfig(level=logging.INFO)



def parse_args():
    parser = argparse.ArgumentParser(description='RetinaFace quantize')
    
    parser.add_argument('--model_path', type=str, default="../model/mobilenet0.25_Final.pth", help="path to onnx model")
    parser.add_argument('--dataset_path', type=str, default="../testbench/images/WIDERVAL", help="path to dataset")
    parser.add_argument('--input_shape', type=tuple, default=(1, 3, 640, 640), help="input shape of acuity model")
    parser.add_argument('-q','--quantize_type', type=str, default="float16", help="quantize data type", choices=['int8', 'uint8', 'float16', 'bfloat16', 'int16'])
    parser.add_argument('--quantize_batch_size', type=int, default=10, help="size of quantize cali data")
    parser.add_argument('--hybrid', type=bool, default=False, help="hybrid quantize or not ")
    parser.add_argument('--output_path', type=str, default="./quantize_file", help="path to save quantize engine")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    print(args)

    
    return args


if __name__ == "__main__":
    args = parse_args()
    # check params

    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # initialize net
    retianface = RetinaFace(args)
    for img_list, filename_list in retianface.DataLoader(args.dataset_path, args.quantize_batch_size):
        logging.info(f"len of img_list: {len(img_list)}")
        # os._exit(0)
        retianface.vsi_quantize_net(img_list, 
                                quantize_type=args.quantize_type,
                                cali_batch_size=args.quantize_batch_size,
                                hybrid=False)
        break

    print("Quantize done.")
    print("quantize file is saved in ./{}_{}.quantize".format(retianface.net_name, args.quantize_type))
    print("Please run infer.py to test the engine file.")
