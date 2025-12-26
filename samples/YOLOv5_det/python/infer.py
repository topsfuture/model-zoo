
import os
import json
import cv2
import argparse
import numpy as np
from utils import COLORS, COCO_CLASSES
from YOLOv5 import YOLOv5


import logging
logging.basicConfig(level=logging.INFO)

def draw_numpy(image, boxes, masks=None, classes_ids=None, class_scores=None):
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        logging.debug("class id={}, score={}, (x1={},y1={},x2={},y2={})".format(classes_ids[idx],class_scores[idx], x1, y1, x2, y2))
        if class_scores[idx] < 0.25:
            continue
        if classes_ids is not None:
            color = COLORS[int(classes_ids[idx]) + 1]
        else:
            color = (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        if classes_ids is not None and class_scores is not None:
            classes_ids = classes_ids.astype(np.int8)
            cv2.putText(image, COCO_CLASSES[classes_ids[idx] + 1] + ':' + str(round(class_scores[idx], 2)),
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)
        if masks is not None:
            mask = masks[:, :, idx]
            image[mask] = image[mask] * 0.5 + np.array(color) * 0.5
        
    return image
   
def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))

        
    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir) 
    
    # initialize net
    yolov5 = YOLOv5(args)
    yolov5.init_model_infer(args)
    
    batch_size = yolov5.batch_size

    # test images
    if os.path.isdir(args.input): 
        img_list = []
        filename_list = []
        results_list = []
        cn = 0
        for root, dirs, filenames in os.walk(args.input):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg','.png','.jpeg','.bmp','.webp']:
                    continue
                img_file = os.path.join(root, filename)
                cn += 1
                logging.info("{}, img_file: {}".format(cn, img_file))
                # decode
                src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
                if src_img is None:
                    logging.error("{} imdecode is None.".format(img_file))
                    continue
                if len(src_img.shape) != 3:
                    src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
                img_list.append(src_img)
                filename_list.append(filename)
                
                if (len(img_list) == batch_size or cn == len(filenames)) and len(img_list):
                    # predict
                    results = yolov5(img_list)  # result: list of ndarray, shape(n,6), element:[x1,y1,x2,y2,scores, class_ids]

                    
                    for i, filename in enumerate(filename_list):
                        print(f"i is : {i}")
                        det = np.array(results[i])
                        # save image
                        # res_img = draw_numpy(img_list[i], det[:,:4], masks=None, classes_ids=det[:, -1], class_scores=det[:, -2])
                        # cv2.imwrite(os.path.join(output_img_dir, filename), res_img)
                        
                        # save result
                        res_dict = dict()
                        res_dict['image_name'] = filename
                        res_dict['bboxes'] = []
                        for idx in range(det.shape[0]):
                            bbox_dict = dict()
                            x1, y1, x2, y2, score, category_id = det[idx]
                            bbox_dict['bbox'] = [float(round(x1, 5)), float(round(y1, 5)), float(round(x2 - x1,5)), float(round(y2 -y1, 5))]
                            bbox_dict['category_id'] = int(category_id)
                            bbox_dict['score'] = float(round(score,5))
                            res_dict['bboxes'].append(bbox_dict)
                        results_list.append(res_dict)
                        
                    img_list.clear()
                    filename_list.clear()
                
        # save results
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        json_name = "yolov5_" + args.quantize_type + "_result.json"
        with open(os.path.join(output_dir, json_name), 'w') as jf:
            # json.dump(results_list, jf)
            json.dump(results_list, jf, indent=4, ensure_ascii=False)
        logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))
        
    # test video
    else:
        cap = cv2.VideoCapture()
        if not cap.open(args.input):
            raise Exception("can not open the video")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # print(fps, size)
        save_video = os.path.join(output_dir, os.path.splitext(os.path.split(args.input)[1])[0] + '.avi')
        out = cv2.VideoWriter(save_video, fourcc, fps, size)
        cn = 0
        frame_list = []
        end_flag = False
        while not end_flag:
            ret, frame = cap.read()
            if not ret or frame is None:
                end_flag = True
            else:
                frame_list.append(frame)
            if (len(frame_list) == batch_size or end_flag) and len(frame_list):
                results = yolov5(frame_list)
                for i, frame in enumerate(frame_list):
                    det = results[i]
                    cn += 1
                    logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                    res_frame = draw_numpy(frame_list[i], det[:,:4], masks=None, classes_ids=det[:, -2], class_scores=det[:, -1])
                    out.write(res_frame)
                frame_list.clear()
        cap.release()
        out.release()
        logging.info("result saved in {}".format(save_video))
        

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../val2017_1000/', help='path sof input')
    
    parser.add_argument('--onnx_path', type=str, default='../yolov5s.onnx', help='path of onnx model')
    parser.add_argument('--input_shape', type=tuple, default=(1, 3, 640, 640), help='input shape of model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-q', '--quantize_type', type=str, default="float16", help="quantize data type",choices=['int8', 'uint8', 'float16', 'bfloat16', 'int16','float32'])
    
    parser.add_argument('--conf_thresh', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.45, help='nms threshold')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')
