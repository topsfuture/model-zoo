
import argparse
import json
import os
import numpy as np
import cv2
from typing import Optional, List, Tuple, Generator
from YOLOv8 import YOLOv8, logger
from quantize import DataLoader
from utils import COCO_CLASSES, COLORS

def save_json(results_list, output_dir):
    # if dataset_path[-1] == '/':
    #     dataset_path = dataset_path[: -1]
    json_name = "yolo8_" + args.quantize_type + "_result.json"
    
    with open(os.path.join(output_dir, json_name), 'w') as jf:
        json.dump(results_list, jf, indent=4, ensure_ascii=False)
    
    
def draw_numpy(image, boxes, masks=None, classes_ids=None, conf_scores=None ):
    " draw boxes in one image"
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()

        if classes_ids is not None:
            color = COLORS[int(classes_ids[idx]) + 1]
        else:
            color = (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        if classes_ids is not None and conf_scores is not None:
            classes_ids = classes_ids.astype(np.int8)
            cv2.putText(image, COCO_CLASSES[classes_ids[idx] + 1] + ':' + str(round(conf_scores[idx], 2)),
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)
        if masks is not None:
            mask = masks[:, :, idx]
            image[mask] = image[mask] * 0.5 + np.array(color) * 0.5
        # logger.debug("class id={}, score={}, (x1={},y1={},x2={},y2={})".format(classes_ids[idx],conf_scores[idx], x1, y1, x2, y2))
    return image

def visual_and_json(filename_list, img_list, outs, output_img_dir):
    results_list = []
    for i, filename in enumerate(filename_list):
        det = outs[i]
        # visualize
        det_draw = det[det[:, -2] > 0.25]
        res_img = draw_numpy(img_list[i],
                                det_draw[:, :4],
                                masks=None,
                                classes_ids=det_draw[:, -1],
                                conf_scores=det_draw[:, -2])
        cv2.imwrite(os.path.join(output_img_dir, filename), res_img)
        
        # save json
        res_dict = {}
        res_dict['image_name'] = filename 
        res_dict['bboxes'] = []
        for idx in range(det.shape[0]):
            bbox_dict = {}
            x1, y1, x2, y2, score, category_id = det[idx]
            bbox_dict['bbox'] = [float(round(x1, 3)), float(round(y1, 3)), float(round(x2 - x1,3)), float(round(y2 -y1, 3))]
            bbox_dict['category_id'] = int(category_id)
            bbox_dict['score'] = float(round(score,5))
            res_dict['bboxes'].append(bbox_dict)
        results_list.append(res_dict)
            
    return results_list


def parse_args():
    
    pass

def check_dir(dir):
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except Exception as e:
            logger.error(f"fail to mkdir {dir}")
            
def main(args):

    
    check_dir(args.output_dir)
    check_dir(args.output_img_dir)
    
    
    # 创建nn, 载入 quantize_net
    yolov8 = YOLOv8(args.onnx_path, args.batch_size)
    if args.quantize_type in ['int8', 'uint8', 'float16', 'bfloat16', 'int16']:
        yolov8.load_q_net(args.quantize_type)
    
    yolov8.nn.build_inference_session(yolov8.acuity_net)
    
    # 推理 DataLoader 生成器
    results_list = []
    for img_list, filename_list in DataLoader(args.dataset_path, args.batch_size):
        outs = yolov8(img_list)
        
        results_list.extend(visual_and_json(filename_list, img_list, outs, args.output_img_dir))

    # save json
    save_json(results_list, args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="yolov8 det")
    parser.add_argument('--onnx_path', type=str, default="../yolov8s.onnx", help="path to onnx model")
    parser.add_argument('--dataset_path', type=str, default="../val2017_1000", help="path to dataset")
    parser.add_argument('-q','--quantize_type', type=str, default="float16", help="quantize data type")
    parser.add_argument('--batch_size', type=int, default=30, help="size of batch")
    parser.add_argument('--output_dir', type=str, default="./output", help="path of output")
    parser.add_argument('--output_img_dir', type=str, default="./output/images", help="path of output image")
    
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
    logger.info("infer done.")
