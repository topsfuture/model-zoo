#!/usr/bin/env python3
"""
板端精度评估脚本 — 读取板端 JSON 输出 + mask PNG，与 COCO GT 对比计算 bbox/segm AP。

用法:
    python coco_eval.py \
        --board-results board_results.json \
        --board-mask-dir ./masks/ \
        --gt-ann instances_val2017.json \
        --output board_precision.txt

前提条件:
    在板端使用 JSON 模式运行推理:
    ./yolo11s_seg_soc -m yolo11s_seg_float16.nb -i input_images/ -o output_images/ \
        --conf 0.001 --nms 0.65 --max-det 300 \
        --json --mask-dir ./masks/
"""

import json
import os
import copy
import argparse
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# YOLO 0-79 (continuous) -> COCO category_id (non-continuous)
# COCO is missing IDs: 12, 26, 29, 30, 45, 66, 68, 69, 71
YOLO_TO_COCO = {
    0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:10, 10:11, 11:13, 12:14, 13:15, 14:16,
    15:17, 16:18, 17:19, 18:20, 19:21, 20:22, 21:23, 22:24, 23:25, 24:27, 25:28, 26:31,
    27:32, 28:33, 29:34, 30:35, 31:36, 32:37, 33:38, 34:39, 35:40, 36:41, 37:42, 38:43,
    39:44, 40:46, 41:47, 42:48, 43:49, 44:50, 45:51, 46:52, 47:53, 48:54, 49:55, 50:56,
    51:57, 52:58, 53:59, 54:60, 55:61, 56:62, 57:63, 58:64, 59:65, 60:67, 61:70, 62:72,
    63:73, 64:74, 65:75, 66:76, 67:77, 68:78, 69:79, 70:80, 71:81, 72:82, 73:84, 74:85,
    75:86, 76:87, 77:88, 78:89, 79:90
}


def fix_category_id(cat_id):
    """Convert YOLO category_id (1-80 continuous) to COCO category_id (non-continuous)."""
    yolo_idx = int(cat_id) - 1  # 1-80 -> 0-79
    return YOLO_TO_COCO.get(yolo_idx, cat_id)


def get_image_id(coco_gt, filename):
    """Get COCO image_id from filename (e.g., '000000000785.jpg' -> 785)."""
    basename = os.path.splitext(os.path.basename(filename))[0]
    try:
        img_id = int(basename)
        # Verify this image_id exists in GT
        if img_id in coco_gt.imgs:
            return img_id
    except ValueError:
        pass
    return None


def mask_png_to_rle(mask_path, img_h, img_w):
    """
    Read mask PNG and convert to RLE format.
    Mask PNG is grayscale, saved by C++ --mask-dir.
    """
    if not os.path.exists(mask_path):
        return None
    
    # Read grayscale mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    # Resize to image dimensions using INTER_NEAREST (no interpolation)
    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    
    # Binarize: >127 is foreground
    mask_bin = (mask > 127).astype(np.uint8)
    
    # Encode to RLE
    rle = maskUtils.encode(np.asfortranarray(mask_bin))
    rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to string
    
    return rle


def filter_coco_gt(coco_gt, valid_img_ids):
    """Filter COCO GT to only include valid image IDs."""
    coco_filtered = copy.deepcopy(coco_gt)
    
    # Filter images
    coco_filtered.imgs = {img_id: img for img_id, img in coco_filtered.imgs.items() 
                          if img_id in valid_img_ids}
    
    # Filter annotations
    valid_ann_ids = [ann['id'] for ann in coco_filtered.anns.values() 
                     if ann['image_id'] in valid_img_ids]
    coco_filtered.anns = {ann_id: ann for ann_id, ann in coco_filtered.anns.items() 
                          if ann_id in valid_ann_ids}
    
    return coco_filtered


def load_board_results(coco_gt, results_json, mask_dir):
    """
    Load board results from JSON and mask PNGs.
    Convert to COCO eval format: list of dicts with image_id, category_id, bbox, score, segmentation.
    """
    with open(results_json, 'r') as f:
        data = json.load(f)
    
    coco_results = []
    
    for item in data['results']:
        filename = item['image']
        img_id = get_image_id(coco_gt, filename)
        
        if img_id is None:
            print(f"Warning: Cannot find image_id for {filename}")
            continue
        
        # Get image dimensions
        img_info = coco_gt.imgs[img_id]
        img_h, img_w = img_info['height'], img_info['width']
        
        for det_idx, det in enumerate(item['detections']):
            # Fix category_id (YOLO 1-80 -> COCO non-continuous)
            cat_id = fix_category_id(det['category_id'])
            
            # Extract bbox [x, y, w, h]
            bbox = det['bbox']
            
            # Extract score
            score = det['score']
            
            # Convert mask PNG to RLE
            mask_file = det['mask_file']
            mask_path = os.path.join(mask_dir, os.path.basename(mask_file))
            segmentation = mask_png_to_rle(mask_path, img_h, img_w)
            
            if segmentation is None:
                print(f"Warning: Cannot load mask {mask_path}")
                continue
            
            coco_results.append({
                'image_id': img_id,
                'category_id': cat_id,
                'bbox': bbox,
                'score': score,
                'segmentation': segmentation
            })
    
    return coco_results


def evaluate(coco_gt, coco_dt, iou_type='bbox'):
    """Run COCO evaluation for bbox or segm."""
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def main():
    parser = argparse.ArgumentParser(description='Board precision evaluation')
    parser.add_argument('--board-results', type=str, required=True,
                        help='Board JSON results from --json mode')
    parser.add_argument('--board-mask-dir', type=str, required=True,
                        help='Directory containing mask PNGs from --mask-dir')
    parser.add_argument('--gt-ann', type=str, required=True,
                        help='COCO GT annotation file (instances_val2017.json)')
    parser.add_argument('--output', type=str, default='board_precision.txt',
                        help='Output report path')
    args = parser.parse_args()
    
    # Load COCO GT
    print(f"Loading GT annotations from {args.gt_ann}")
    coco_gt = COCO(args.gt_ann)
    
    # Load board results
    print(f"Loading board results from {args.board_results}")
    print(f"Loading masks from {args.board_mask_dir}")
    coco_results = load_board_results(coco_gt, args.board_results, args.board_mask_dir)
    print(f"Loaded {len(coco_results)} detections")
    
    if len(coco_results) == 0:
        print("Error: No valid detections loaded")
        return
    
    # Create COCO DT from results
    coco_dt = coco_gt.loadRes(coco_results)
    
    # Filter GT to only include images with detections
    valid_img_ids = set([r['image_id'] for r in coco_results])
    coco_gt_filtered = filter_coco_gt(coco_gt, valid_img_ids)
    print(f"Filtered GT to {len(coco_gt_filtered.imgs)} images")
    
    # Run evaluation
    print("\n" + "="*60)
    print("BBox Evaluation")
    print("="*60)
    bbox_eval = evaluate(coco_gt_filtered, coco_dt, iou_type='bbox')
    
    print("\n" + "="*60)
    print("Segm Evaluation")
    print("="*60)
    segm_eval = evaluate(coco_gt_filtered, coco_dt, iou_type='segm')
    
    # Write report
    with open(args.output, 'w') as f:
        f.write("Board Precision Evaluation Report\n")
        f.write("="*60 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Board results: {args.board_results}\n")
        f.write(f"  Mask directory: {args.board_mask_dir}\n")
        f.write(f"  GT annotations: {args.gt_ann}\n")
        f.write(f"  Images evaluated: {len(valid_img_ids)}\n")
        f.write(f"  Total detections: {len(coco_results)}\n\n")
        
        f.write("BBox Metrics:\n")
        f.write(f"  AP@0.50:0.95: {bbox_eval.stats[0]:.4f}\n")
        f.write(f"  AP@0.50:      {bbox_eval.stats[1]:.4f}\n")
        f.write(f"  AP@0.75:      {bbox_eval.stats[2]:.4f}\n")
        f.write(f"  AR@0.50:0.95: {bbox_eval.stats[6]:.4f}\n\n")
        
        f.write("Segm Metrics:\n")
        f.write(f"  AP@0.50:0.95: {segm_eval.stats[0]:.4f}\n")
        f.write(f"  AP@0.50:      {segm_eval.stats[1]:.4f}\n")
        f.write(f"  AP@0.75:      {segm_eval.stats[2]:.4f}\n")
        f.write(f"  AR@0.50:0.95: {segm_eval.stats[6]:.4f}\n")
    
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
