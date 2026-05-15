
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
       


def validation_process(args):
    jsonfile = args.output
    all_results = [] # 确保在函数开头定义
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
                    results1, results2, results3 = retinaface(img_list)
                    for i, fname in enumerate(filename_list):
                        #print(f"aaa file name {fname}")                        
                        img_entry = {
                            "bboxes": [],
                            "image_name": fname
                        }
                        for j in range(results1[i].shape[0]):
                            box = results1[i][j]
                            landmark = results2[i][j]
                            score = results3[i][j]

                            # --- 修改处 1：使用 .tolist() 和 float() ---
                            img_entry["bboxes"].append({
                                "bbox": box[:4].tolist(),
                                "landmarks": landmark.tolist(),
                                "score": float(score)
                            })                            
                            # ---------------------------------------

                            #print(f"score aaa is {score}")
                            #draw_one(box, landmark, img_list[i], label="{}:{:.2f}".format('Face',box[4]))
                        #cv2.imwrite(os.path.join(output_dir, fname), img_list[i])                           
                        all_results.append(img_entry)                        

                    img_list.clear()
                    filename_list.clear()

            # process remaining images
            if img_list:
                results1, results2, results3 = retinaface(img_list)

                for i, fname in enumerate(filename_list):
                    #print(f"bbb file name {fname}")
                    img_entry = {
                        "bboxes": [],
                        "image_name": fname
                    }

                    for j in range(results1[i].shape[0]):
                        box = results1[i][j]
                        landmark = results2[i][j]
                        score = results3[i][j]

                        # --- 修改处 2：使用 .tolist() 和 float() ---
                        img_entry["bboxes"].append({
                            "bbox": box[:4].tolist(),
                            "landmarks": landmark.tolist(),
                            "score": float(score)
                        })    
                        # ---------------------------------------

                        #print(f"score bbb is {score}")
                        #draw_one(box, landmark, img_list[i], label="{}:{:.2f}".format('Face',box[4]))
                    #cv2.imwrite(os.path.join(output_dir, fname), img_list[i])
                    all_results.append(img_entry)
                img_list.clear()
                filename_list.clear()
        
        # Save final JSON results
        '''
        with open('output_consistent.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        '''
        print(f"jsonfile {jsonfile}")
        with open(jsonfile, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        
        print(f'Inference results saved to {jsonfile}')
    # test video
    else:
        print("validation not support")



def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../testbench/images/WIDERVAL', help='path of input')
    # parser.add_argument('--input', type=str, default='../testbench/images/face', help='path sof input')
    
    parser.add_argument('--model_path', type=str, default='../model/retinaface.onnx', help='path of caffe model')
    parser.add_argument('--validation', type=bool, default=True, help='validation flag')
    parser.add_argument('--output', type=str, default='output.json', help='output json file name')
    parser.add_argument('--input_shape', type=tuple, default=(1, 3, 640, 640), help='input shape of model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-q', '--quantize_type', type=str, default="float16", help="quantize data type",choices=['int8', 'uint8', 'float16', 'bfloat16', 'int16','float32'])
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    validation_process(args)
    print('all done.')


