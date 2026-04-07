
import os
import json
import cv2
import argparse
import numpy as np
from OpenPose import OpenPose


import logging
logging.basicConfig(level=logging.INFO)
   
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
    openpose = OpenPose(args)
    openpose.init_model_infer(args)
    
    batch_size = openpose.batch_size

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
                    results = openpose(img_list)  # result: list of ndarray, shape(n,6), element:[x1,y1,x2,y2,scores, class_ids]

                    
                    for i, filename in enumerate(filename_list):
                        candidate, subset = results[i]
                        # res_img = openpose.draw_pose(img_list[i], candidate, subset)
                        # cv2.imwrite(os.path.join(output_img_dir, filename), res_img)

                        res_dict = dict()
                        res_dict['image_name'] = filename
                        res_dict['keypoints'] = []
                        for n in range(len(subset)):
                            for m in range(openpose.point_num):
                                index = int(subset[n][m])
                                if index == -1:
                                    x, y, score = 0, 0, 0.0
                                else:
                                    x, y, score = candidate[index][0:3]
                                res_dict['keypoints'].append(x)
                                res_dict['keypoints'].append(y)
                                res_dict['keypoints'].append(score)
                        results_list.append(res_dict)
                        
                    img_list.clear()
                    filename_list.clear()
                
        # save results
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        json_name = "openpose" + args.quantize_type + "_result.json"
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
    
    parser.add_argument('--model_path', type=str, default='../pose_iter_440000.prototxt', help='path of caffe model')
    parser.add_argument('--model_weight', type=str, default='../pose_iter_440000.caffemodel', help='weights path of caffe model')
    parser.add_argument('--input_shape', type=tuple, default=(1, 3, 368, 368), help='input shape of model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-q', '--quantize_type', type=str, default="float16", help="quantize data type",choices=['int8', 'uint8', 'float16', 'bfloat16', 'int16','float32'])
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')
