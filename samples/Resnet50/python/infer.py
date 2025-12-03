import json
import cv2
import numpy as np
import logging
import os
import argparse

from Resnet import Resnet
from typing import Generator, Tuple, List, Optional

def save_json(results_list: list, output_dir: str, quantize_type:str,  dataset_path: str):
    if dataset_path[-1] == "/":
        dataset_path = dataset_path[:-1]
    json_name = os.path.split("resnet.onnx")[-1] + '_' + quantize_type + "_" + os.path.split(dataset_path)[-1] + "_python_result.json"
    with open(os.path.join(output_dir, json_name), "w") as jf:
        json.dump(results_list, jf, indent=4, ensure_ascii=False)
    logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))

def visualize_and_json(outputs, img_list:list, filename_list:list, output_img_dir: str)->json: 
    results_list = []
    for i, filename in enumerate(filename_list):
        res_dict = dict()
        logging.info(f"filename: {filename}, res: {outputs[i]}")
        res_dict['filename'] = filename
        res_dict['prediction'] = outputs[i][0]
        res_dict['score'] = outputs[i][1]
        results_list.append(res_dict)
        
        img = img_list[i]
        cv2.putText(img, f"{outputs[i][0]}-{outputs[i][1]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        output_img_path = os.path.join(output_img_dir, filename)
        cv2.imwrite(output_img_path, img)
        
    return results_list
    
    

def DataLoader(dataset_path:str, bs:int) -> Generator[Tuple[List[np.ndarray]], None, None]:
    def decode_image(img_path: str) -> Optional[np.ndarray]:
        try:
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
    if os.path.isdir(dataset_path):
        img_list, filename_list = [], []
        cn = 0
        for path, dirs, filenames in os.walk(dataset_path):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                img_file_path = os.path.join(path, filename)
                cn += 1
                logging.info(f"{cn}, img_file: {img_file_path}")
                src_img = decode_image(img_file_path)
                img_list.append(src_img)
                filename_list.append(filename)
                
                if len(img_list) == bs:
                    yield img_list, filename_list
                    img_list, filename_list = [], []
        if len(img_list) > 0:
            yield img_list, filename_list
        

def parse_args():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument("--onnx_path", type=str, default="../resnet50.onnx", help="path of onnx model")
    parser.add_argument("--dataset_path", type=str, default="../datasets/imagenet_val_1k/img", help="path of dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
    parser.add_argument("--input_shape", type=Tuple[int, int, int, int], default=(1, 3, 224, 224), help="input shape of model")
    parser.add_argument("--conf_thresh", type=float, default=0.001, help="Confidence threshold for postprocess")
    parser.add_argument("--nms_thresh", type=float, default=0.6, help="NMS threshold for postprocess")
    parser.add_argument('-q',"--quantize_type", type=str, default="float16", choices=['int8', 'int16', 'float16', 'bfloat16', 'uint8','float'], help="Quantize type")
    args = parser.parse_args()
    return args

def main():
    pass
    args = parse_args()
    resnet = Resnet(args.onnx_path, args.input_shape, args.conf_thresh, args.nms_thresh, args.quantize_type)
    if args.quantize_type in ['int8', 'uint8', 'float16', 'bfloat16', 'int16']:
        resnet.load_q_net(args.quantize_type)
    
    resnet.nn.build_inference_session(resnet.acuity_net)
    
    output_dir = "./results"
    output_img_dir = os.path.join(output_dir, "images")
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results_list = []
    for img_list, filename_list in DataLoader(args.dataset_path, args.batch_size):
        outputs = resnet.infer_image(img_list)
        # postprocess(outputs)
        # logging.info(f"outputs shape: {outputs.shape}")
        
        results_list_batch = visualize_and_json(outputs, img_list, filename_list, output_img_dir)
        results_list.extend(results_list_batch)
        img_list, filename_list = [], []
    
    save_json(results_list, output_dir, args.quantize_type ,args.dataset_path)

if __name__ == "__main__":
    main()
    logging.basicConfig(level=logging.INFO)
    logging.info("Done.")