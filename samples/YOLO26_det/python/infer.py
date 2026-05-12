import argparse
import json
import os

import numpy as np
import cv2
from YOLO26 import yolo26, logger
from utils import COCO_CLASSES, COLORS


def iter_images(dataset_dir):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    for root, _, filenames in os.walk(dataset_dir):
        for filename in filenames:
            if os.path.splitext(filename)[-1].lower() not in ['.jpg', '.png', '.jpeg', '.bmp', '.webp']:
                continue
            img_path = os.path.join(root, filename)
            src_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            if src_img is None:
                continue
            if len(src_img.shape) != 3:
                src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
            yield src_img, filename


def draw_numpy(image, boxes, classes_ids=None, conf_scores=None):
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        color = COLORS[int(classes_ids[idx]) + 1] if classes_ids is not None else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        if classes_ids is not None and conf_scores is not None:
            cls_id = int(classes_ids[idx])
            label = f"{COCO_CLASSES[cls_id + 1]}:{conf_scores[idx]:.2f}"
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)
    return image


def visual_and_json(filename, src_img, det, output_img_dir):
    
    if det is None:
        return
    logger.debug(f"det shape: {det.shape}")
    res_img = draw_numpy(src_img, det[:, :4],
                         classes_ids=det[:, 5],
                         conf_scores=det[:, 4])
    cv2.imwrite(os.path.join(output_img_dir, filename), res_img)

    res_dict = {"image_name": filename, "bboxes": []}
    for row in det:
        x1, y1, x2, y2, score, category_id = row
        res_dict["bboxes"].append({
            "bbox": [round(float(x1), 3), round(float(y1), 3),
                     round(float(x2 - x1), 3), round(float(y2 - y1), 3)],
            "category_id": int(category_id),
            "score": round(float(score), 5),
        })
    return res_dict


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_img_dir, exist_ok=True)

    model = yolo26(args)
    if args.quantize_type in ['int8', 'uint8', 'float16', 'bfloat16', 'int16']:
        model.load_q_net(args.quantize_type)

    model.nn.build_inference_session(model.acuity_net)

    results_list = []
    for src_img, filename in iter_images(args.dataset_path):
        det = model(src_img)
        results_list.append(visual_and_json(filename, src_img, det, args.output_img_dir))

    json_name = f"yolo26_{args.quantize_type}_result.json"
    with open(os.path.join(args.output_dir, json_name), 'w') as jf:
        json.dump(results_list, jf, indent=4, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="yolo det")
    parser.add_argument('--onnx_path', type=str, default="../models/yolo26s.onnx")
    parser.add_argument('--dataset_path', type=str, default="/home/intchains/dev-space/yolov26/datasets/val2017_1000/")
    parser.add_argument('-q', '--quantize_type', type=str, default="float32")
    parser.add_argument('--conf_thresh', type=float, default=0.001)
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--output_img_dir', type=str, default="./output/images")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
    logger.info("infer done.")