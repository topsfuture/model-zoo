
import numpy as np
import cv2
from typing import Optional, Generator, Tuple, List
from YOLOv8 import YOLOv8, logger
import os
import argparse


def DataLoader(dataset_path:str, batch_size:int) -> Generator[Tuple[List[np.ndarray]],None, None]:
    def decode_image(img_path: str) -> Optional[np.ndarray]:
        try:
            src_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            
            if src_img is None:
                logger.error(f"Failed to decode image: {img_path}")
                return None
            # if len(src_img.shape) == 1:  # GRAY 
            if len(src_img.shape) != 3:
                src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
            if len(src_img.shape) == 4: # alpha channel
                pass
            
            logger.debug(f"decode image: {img_path} \t shape:{src_img.shape}")
            return src_img
            
        except Exception as e:
            logger.error(f"Failed to decode image: {img_path}")
            return None
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("dataset path is not exist")

    img_list, filename_list = [], []
    
    for path, dirs, filenames in os.walk(dataset_path):
        for filename in filenames:
            if os.path.splitext(filename)[-1].lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            img_path = os.path.join(path, filename)
            src_img = decode_image(img_path)
            
            img_list.append(src_img)
            filename_list.append(filename)
            
            if len(img_list) == batch_size:
                yield img_list, filename_list
                img_list.clear()  # 下一batch 清空
                filename_list.clear()
        
        if len(img_list) > 0:
            yield img_list, filename_list
            img_list.clear()
            filename_list.clear()
        



def main(args):
    
    # create nn
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError()
    
    yolov8 = YOLOv8(args.onnx_path)
    
    # quantize
    for img_list, filename_list in DataLoader(args.dataset_path, args.quantize_batch_size):
        yolov8.vsi_quantize_net(img_list, 
                                args.quantize_type,
                                len(img_list),
                                args.hybrid)
        break

    


def parse_args():
    parser = argparse.ArgumentParser(description="yolov8 det quantize")
    parser.add_argument('--onnx_path', type=str, default="../yolov8s.onnx", help="path to onnx model")
    parser.add_argument('--dataset_path', type=str, default="../val2017_1000", help="path to dataset")
    parser.add_argument('-q','--quantize_type', type=str, default="float16", help="quantize data type")
    parser.add_argument('--quantize_batch_size', type=int, default=10, help="size of quantize cali data")
    parser.add_argument('--hybrid', type=bool, default=False, help="hybrid quantize or not ")
    
    
    args = parser.parse_args()
    return args    


if __name__ == "__main__":
    
    args = parse_args()
    main(args)
    logger.info("quantize done")
