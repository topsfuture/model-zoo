import logging
import argparse
import os

from Resnet import Resnet
from typing import Generator, Tuple, List, Optional
import numpy as np
import cv2



def decode_image(img_path: str) -> Optional[np.ndarray]:
    try:
        # img = cv2.imread(img_path)
        src_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if src_img is None:
            logging.error(f"{img_path} imread is None.")
            return None
        if len(src_img.shape) !=3:
            src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
        return src_img
    except Exception as e:
        logging.error(f"Failed to decode image:{img_path}")
        return None


def DataLoader(dataset_path:str , batch_size:int, ) -> Generator[Tuple[List[np.ndarray], List[str]], None, None]:
    if os.path.isdir(dataset_path):
        img_list, filename_list = [], []
        for path, dirs, filenames in os.walk(dataset_path):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                img_path = os.path.join(path, filename)
                # img = cv2.imread(img_path)
                img = decode_image(img_path)
                if img is None:
                    continue
                img_list.append(img)
                filename_list.append(filename)
                if len(img_list) == batch_size:
                    yield img_list, filename_list
                    img_list, filename_list = [], []
        if len(img_list) > 0:
            yield img_list, filename_list
        
        
        
        
    else:
        raise FileNotFoundError(f"{dataset_path} is not a directory.")



def parse_args():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument("--onnx_path", type=str, default="../resnet50.onnx", help="path of onnx model")
    parser.add_argument("--input_shape", type=Tuple[int, int, int, int], default=(1, 3, 224, 224), help="input shape of model")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="Confidence threshold for postprocess")
    parser.add_argument("--nms_thresh", type=float, default=0.25, help="NMS threshold for postprocess")
    parser.add_argument("--dataset_path", type=str, default="../datasets/imagenet_val_1k/img", help="path of input, must be image directory")
    parser.add_argument("--batch_size", type=int, default=5, help="batch size")
    parser.add_argument('-q',"--quantize_type", type=str, default="uint8", choices=["int8", "uint8", "int16"], help="quantize data type, select from ['int8', 'uint8', 'int16']")
    parser.add_argument("--path_to_quantize_file", type=Optional[str], default=None, help="path to .quantize file")
    parser.add_argument("--hybrid", type=bool, default=False, help="enable hybrid quantize")
    
    args = parser.parse_args()
    return args

def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError("{dataset_path} does not exist.")
    
    resnet = Resnet(args.onnx_path, args.input_shape, args.conf_thresh, args.nms_thresh, args.quantize_type)
    if args.path_to_quantize_file is None:
        for img_list, filename_list in DataLoader(args.dataset_path, args.batch_size):
            resnet.vsi_nn_quantize_net(img_list,
                                       args.quantize_type,
                                       resnet.net_name,
                                       args.onnx_path,
                                       len(img_list),
                                       args.hybrid)
            break
            
        
    else:
        logging.info(f"Quantize file already exists:{args.path_to_quantize_file}")

if __name__ == "__main__":
    main()
    logging.info("quantize success.")