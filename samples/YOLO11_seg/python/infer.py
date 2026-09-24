import argparse
import json
import os
import time
import numpy as np
import cv2
import onnxruntime as ort

from postprocess_numpy import PostProcess
from utils import COCO_CLASSES, COLORS


def draw_masks(image, boxes, masks, class_ids, scores):
    """Draw boxes and instance masks on image."""
    h, w = image.shape[:2]
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(np.int32).tolist()
        cid = int(class_ids[i])
        color = COLORS[cid % len(COLORS)]

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        label = f"{COCO_CLASSES[cid]}:{round(scores[i], 2)}"
        cv2.putText(image, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness=2)

        # Draw mask overlay
        if masks.shape[2] > i:
            mask = masks[:, :, i]
            color_mask = np.array(color, dtype=np.uint8)
            image[mask] = (image[mask].astype(np.float32) * 0.5 +
                          color_mask.astype(np.float32) * 0.5).astype(np.uint8)
    return image


def preprocess(image, input_size=640):
    """
    Preprocess: letterbox resize to 640x640, BGR->RGB, /255, HWC->CHW
    Matches C++ preprocess_image when NB has preprocess node.
    """
    h, w = image.shape[:2]

    # Letterbox resize
    ratio = min(input_size / h, input_size / w)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    pad_left = (input_size - new_w) // 2
    pad_top = (input_size - new_h) // 2
    pad_right = input_size - new_w - pad_left
    pad_bottom = input_size - new_h - pad_top

    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # BGR -> RGB, /255, HWC -> CHW
    img = padded[:, :, ::-1].astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis, ...]

    return img


def inference_onnxruntime(onnx_path, image, conf_thresh=0.25, nms_thresh=0.45, max_det=300):
    """Run inference with ONNX Runtime."""
    # Load model
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    # Preprocess
    h, w = image.shape[:2]
    input_tensor = preprocess(image, input_size=640)

    # Inference
    start = time.time()
    outputs = sess.run(None, {input_name: input_tensor})
    infer_time = time.time() - start

    # outputs[0]: (1, 116, 8400) detections
    # outputs[1]: (1, 32, 160, 160) protos
    preds = outputs[0]
    protos = outputs[1]

    # Postprocess
    postprocess = PostProcess(
        conf_thresh=conf_thresh,
        nms_thresh=nms_thresh,
        max_det=max_det,
        mask_thresh=0.5
    )

    result = postprocess(preds, protos, (640, 640), (w, h))

    return result, infer_time


def inference_acuity(onnx_path, image, conf_thresh=0.25, nms_thresh=0.45, max_det=300):
    """Run inference with acuity (VSInn backend)."""
    from YOLO11Seg import YOLO11Seg

    # Load model
    yolo = YOLO11Seg(
        onnx_path,
        batch_size=1,
        conf_thresh=conf_thresh,
        nms_thresh=nms_thresh,
        mask_thresh=0.5
    )
    yolo.nn.build_inference_session(yolo.acuity_net)

    # Inference
    start = time.time()
    result = yolo([image])[0]
    infer_time = time.time() - start

    return result, infer_time


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11-seg inference")
    parser.add_argument('--onnx_path', type=str, required=True,
                        help="path to ONNX model")
    parser.add_argument('--image', type=str, required=True,
                        help="input image path")
    parser.add_argument('--output', type=str, default='output.jpg',
                        help="output image path")
    parser.add_argument('--conf', type=float, default=0.25,
                        help="confidence threshold")
    parser.add_argument('--nms', type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument('--max-det', type=int, default=300,
                        help="max detections per image")
    parser.add_argument('--backend', type=str, default='onnxruntime',
                        choices=['onnxruntime', 'acuity'],
                        help="inference backend (default: onnxruntime)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: cannot load image {args.image}")
        return

    print(f"Backend: {args.backend}")
    print(f"Image: {args.image} ({image.shape[1]}x{image.shape[0]})")

    # Inference
    if args.backend == 'onnxruntime':
        result, infer_time = inference_onnxruntime(
            args.onnx_path, image, args.conf, args.nms, args.max_det
        )
    else:
        result, infer_time = inference_acuity(
            args.onnx_path, image, args.conf, args.nms, args.max_det
        )

    boxes = result['boxes']
    scores = result['scores']
    class_ids = result['class_ids']
    masks = result['masks']

    print(f"Detected {len(boxes)} objects in {infer_time*1000:.1f}ms")

    # Print detections
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cid = int(class_ids[i])
        score = scores[i]
        print(f"  [{i}] {COCO_CLASSES[cid]}: {score:.3f} @ ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    # Draw and save
    vis_image = draw_masks(image.copy(), boxes, masks, class_ids, scores)
    cv2.imwrite(args.output, vis_image)
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
