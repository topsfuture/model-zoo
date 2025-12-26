
import argparse
import json
import logging
import os
logging.basicConfig(level=logging.DEBUG)

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--gt_path', type=str, default='../../datasets/imagenet_val_1k/label.txt', help='path of label')
    parser.add_argument('-r','--result_json', type=str, default='resnet.onnx_float_img_python_result.json', help='path of result json')
    args = parser.parse_args()
    return args

def main(args):

    d_gt = {}
    with open(args.gt_path, 'r') as f:
        for line in f:
            path, label = line.strip().split()
            filename = os.path.basename(path)   # 去掉路径，只保留文件名
            d_gt[filename] = int(label)
    
    d_pred = {}
    with open(args.result_json, 'r') as f:
        res_json = json.load(f)
    for res in res_json:
        d_pred[res['filename'].split('/')[-1]] = res['prediction']

    correct = 0
    for k, gt in d_gt.items():
        prediction = d_pred[k]
        if int(gt)==prediction:
            correct += 1
    acc = correct / float(len(d_gt))

    logging.info('gt_path: {}'.format(args.gt_path))
    logging.info('pred_path: {}'.format(args.result_json))
    logging.info('ACC: {:.5f}%'.format(acc*100))


if __name__ == '__main__':
    args = argsparser()
    main(args)
