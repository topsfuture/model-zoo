import numpy as np
import cv2
from typing import Optional, Generator, Tuple, List
import os
import argparse

from YOLO11Seg import YOLO11Seg, logger


def DataLoader(dataset_path: str, batch_size: int
               ) -> Generator[Tuple[List[np.ndarray], List[str]], None, None]:
    """Load images from dataset_path in batches."""

    def decode_image(img_path: str) -> Optional[np.ndarray]:
        try:
            src_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            if src_img is None:
                logger.error(f"Failed to decode image: {img_path}")
                return None
            if len(src_img.shape) != 3:
                src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
            return src_img
        except Exception as e:
            logger.error(f"Failed to decode image: {img_path}")
            return None

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    img_list, filename_list = [], []

    for path, dirs, filenames in os.walk(dataset_path):
        for filename in filenames:
            if os.path.splitext(filename)[-1].lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            img_path = os.path.join(path, filename)
            src_img = decode_image(img_path)
            if src_img is None:
                continue
            img_list.append(src_img)
            filename_list.append(filename)
            if len(img_list) == batch_size:
                yield img_list, filename_list
                img_list.clear()
                filename_list.clear()

    if len(img_list) > 0:
        yield img_list, filename_list
        img_list.clear()
        filename_list.clear()


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11-seg quantize")
    parser.add_argument('--onnx_path', type=str,
                        default="../models/yolo11s-seg.onnx",
                        help="path to onnx model")
    parser.add_argument('--dataset_path', type=str,
                        default="../datasets/coco_val_1000",
                        help="path to calibration dataset")
    parser.add_argument('-q', '--quantize_type', type=str,
                        default="float16",
                        help="quantize data type: int8/uint8/float16/bfloat16/int16")
    parser.add_argument('--quantize_batch_size', type=int,
                        default=10,
                        help="batch size for calibration")
    parser.add_argument('--hybrid', type=bool, default=False,
                        help="use hybrid quantization")
    return parser.parse_args()


def main(args):
    # Create model
    yolo = YOLO11Seg(args.onnx_path, batch_size=1)

    # Run quantization
    for img_list, filename_list in DataLoader(args.dataset_path, args.quantize_batch_size):
        yolo.vsi_quantize_net(
            img_list,
            args.quantize_type,
            len(img_list),
            args.hybrid
        )
        break  # only one batch needed for calibration


if __name__ == "__main__":
    args = parse_args()
    main(args)
    logger.info("Quantization done.")
