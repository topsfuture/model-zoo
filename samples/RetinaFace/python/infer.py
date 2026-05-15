
import os
import json
import cv2
import argparse
import numpy as np
from RetinaFace import RetinaFace
import random

import logging
logging.basicConfig(level=logging.INFO)

def draw_one(box, landmark, img, color = None, label = None, line_thickness = None):

        tl = (
                line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1
        )  

        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

        cv2.circle(img, (int(landmark[0]), int(landmark[1])), 1, (0, 0, 255), 4)
        cv2.circle(img, (int(landmark[2]), int(landmark[3])), 1, (0, 255, 255), 4)
        cv2.circle(img, (int(landmark[4]), int(landmark[5])), 1, (255, 0, 255), 4)
        cv2.circle(img, (int(landmark[6]), int(landmark[7])), 1, (0, 255, 0), 4)
        cv2.circle(img, (int(landmark[8]), int(landmark[9])), 1, (255, 0, 0), 4)

        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                img,
                label,
                (c1[0], c1[1] - 2),
                0,
                tl / 3,
                [225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))

        
    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # initialize net
    retinaface = RetinaFace(args)
    retinaface.create_acuity_net()
    retinaface.init_model_infer(args)
    
    batch_size = retinaface.batch_size

    # test images
    if os.path.isdir(args.input):
        cn = 0
        chunk_size = 4
        img_list = []
        filename_list = []
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

                if len(img_list) == chunk_size:
                    # predict
                    results1, results2,result3 = retinaface(img_list)
                    for i, fname in enumerate(filename_list):
                        for j in range(results1[i].shape[0]):
                            box = results1[i][j]
                            landmark = results2[i][j]
                            draw_one(box, landmark, img_list[i], label="{}:{:.2f}".format('Face',box[4]))
                        cv2.imwrite(os.path.join(output_dir, fname), img_list[i])
                    img_list.clear()
                    filename_list.clear()

            # process remaining images
            if img_list:
                results1, results2,result3 = retinaface(img_list)
                for i, fname in enumerate(filename_list):
                    for j in range(results1[i].shape[0]):
                        box = results1[i][j]
                        landmark = results2[i][j]
                        draw_one(box, landmark, img_list[i], label="{}:{:.2f}".format('Face',box[4]))
                    cv2.imwrite(os.path.join(output_dir, fname), img_list[i])
                img_list.clear()
                filename_list.clear()
        
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
    parser.add_argument('--input', type=str, default='../testbench/images/WIDERVAL', help='path of input')
    # parser.add_argument('--input', type=str, default='../testbench/images/face', help='path sof input')
    
    parser.add_argument('--model_path', type=str, default='../model/retinaface.onnx', help='path of caffe model')
    parser.add_argument('--input_shape', type=tuple, default=(1, 3, 640, 640), help='input shape of model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-q', '--quantize_type', type=str, default="float16", help="quantize data type",choices=['int8', 'uint8', 'float16', 'bfloat16', 'int16','float32'])
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')
