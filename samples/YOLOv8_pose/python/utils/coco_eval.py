from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import json
import argparse


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def convert_to_coco_keypoints(json_file, cocoGt):
    keypoints_openpose_map = ["nose", "Neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist",  \
                        "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"]
    temp_json = []
    images_list = cocoGt.dataset["images"]
    with open(json_file, 'r') as f:
        res_json = json.load(f)
    for res in res_json:
        image_name = res["image_name"]
        keypoints = res["keypoints"]
        # for i in range(2, len(keypoints), 3):
        #     keypoints[i] *= 2
        if res["score"] < 0.5:
            continue

        for image in images_list:
            if image_name == image["file_name"]:
                image_id = image["id"]
                break
        data = dict()
        data['image_id'] = int(image_id)
        data['category_id'] = 1
        data['keypoints'] = keypoints
        score_list = []
        for i in range(int(len(keypoints) / 3)):
            score = keypoints[i * 3 + 2]
            score_list.append(score)
        data['keypoints'] = keypoints
        data['score'] = sum(score_list)/len(score_list) + res["score"]
        temp_json.append(data)
        
    with open('converted.json', 'w') as fid:
        json.dump(temp_json, fid)

def convert_to_coco_bbox(json_file, cocoGt):
    temp_json = []
    coco91class = coco80_to_coco91_class()
    images_list = cocoGt.dataset["images"]
    with open(json_file, 'r') as f:
        res_json = json.load(f)
    for res in res_json:
        image_name = res["image_name"]
        bboxes = res["bboxes"]
        if len(bboxes) == 0:
            continue
        for image in images_list:
            if image_name == image["file_name"]:
                image_id = image["id"]
                break
            
        for i in range(len(bboxes)):
            data = dict()
            data['image_id'] = int(image_id)
            data['category_id'] = coco91class[bboxes[i]['category_id']]
            data['bbox'] = bboxes[i]['bbox']
            data['score'] = bboxes[i]['score']
            temp_json.append(data)
            
    with open('converted.json', 'w') as fid:
        json.dump(temp_json, fid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--ground_truth", type=str, help="Assign the groud true path.", default='instances_val2017_1000.json')
    parser.add_argument("-r", "--result", type=str, help="Assign the detection result path.", default=None)
    parser.add_argument('--ann_type', type=str, default='keypoints', help='type of evaluation')
    args = parser.parse_args()

    cocoGt = COCO(args.ground_truth)    
    if args.ann_type == 'keypoints':
        convert_to_coco_keypoints(args.result, cocoGt)
    if args.ann_type == 'bbox':
        convert_to_coco_bbox(args.result, cocoGt)


    cocoRes = cocoGt.loadRes('converted.json')
    #cocoRes = cocoGt.loadRes(args.result)
    cocoEval = COCOeval(cocoGt, cocoRes, args.ann_type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    