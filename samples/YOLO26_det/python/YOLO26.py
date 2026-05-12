import os
import logging

import numpy as np
import cv2
from acuitylib.vsi_nn import VSInn

def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(' %(levelname)s %(name)s Line:%(lineno)s: %(message)s |  - %(asctime)s ')
    handler.setFormatter(formatter)
    if logger.hasHandlers:
        logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = init_logger()


class yolo26:
    def __init__(self, args):
        self.net_name = "yolo26s"
        self.onnx_path = args.onnx_path
        self.conf_thresh = args.conf_thresh
        self.net_h = 640
        self.net_w = 640
        self.create_nn()

    def create_nn(self):
        self.nn = VSInn()
        json_path = f"./{self.net_name}.json"
        data_path = f"./{self.net_name}.data"

        if os.path.exists(json_path) and os.path.exists(data_path):
            logger.info("Load .json & .data file")
            self.acuity_net = self.nn.create_net()
            self.nn.load_model(self.acuity_net, json_path)
            self.nn.load_model_data(self.acuity_net, data_path)
        else:
            self.acuity_net = self.nn.load_onnx(
                self.onnx_path,
                inputs="images",
                outputs="output0",
                input_size_list="3, 640, 640",
            )
            self.nn.save_model(self.acuity_net, json_path)
            self.nn.save_model_data(self.acuity_net, data_path)

    def load_q_net(self, quantize_type):
        logger.info(f"quantize type: {quantize_type}")
        valid_types = ['int8', 'uint8', 'float16', 'bfloat16', 'int16', 'float32']
        if quantize_type not in valid_types:
            raise ValueError(f"Wrong quantize type: {quantize_type}. Must be one of {valid_types}")

        # q_path = f"./{self.net_name}_{quantize_type}.quantize"
        q_path = f"/home/intchains/dev-space/yolov26/acuity_temp/model_{quantize_type}.quantize"
        if os.path.exists(q_path):
            logger.info(f"Load {quantize_type} quantize file.")
            self.nn.load_model_quantize(self.acuity_net, q_path)
        else:
            raise FileNotFoundError(f"{q_path} does not exist. Please run quantize.py first.")

    def preprocess(self, ori_img):
        """
        Args:
            ori_img: (H, W, 3) BGR ndarray

        Returns:
            img:   (3, 640, 640) float32 ndarray, normalized
            ratio: (ratio_x, ratio_y)
            txy:   (tx, ty) padding offset
        """
        letterbox_img, ratio, (tx, ty) = self._letterbox(ori_img)
        img = letterbox_img.transpose((2, 0, 1))[::-1]  # HWC BGR -> CHW RGB
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img, ratio, (tx, ty)

    def _letterbox(self, im):
        shape = im.shape[:2]
        new_shape = (self.net_h, self.net_w)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        dw = (new_shape[1] - new_unpad[0]) / 2
        dh = (new_shape[0] - new_unpad[1]) / 2

        if (shape[1], shape[0]) != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(264, 264, 264))

        return im, r, (left, top)
    
    def postprocess(self, pred, ori_size, ratio, txy):
        det = pred[pred[:, 4] > self.conf_thresh]
        det[:,[0,2]] -= txy[0]
        det[:,[1,3]] -= txy[1]
        det[:,:4] /= ratio
        det[:, [0, 2]] = det[:, [0, 2]].clip(0, ori_size[0] - 1)  # x1, x2
        det[:, [1, 3]] = det[:, [1, 3]].clip(0, ori_size[1] - 1)  # y1, y2
        return det

    @staticmethod
    def _np_to_tf(img):
        import tensorflow as tf
        return tf.convert_to_tensor(np.expand_dims(img, axis=0))

    def vsi_quantize_net(self, img_list, quantize_type, cali_batch_size, hybrid=False):
        q_er_table = {
            "uint8": "asymmetric_affine",
            "int8": "perchannel_symmetric_affine",
            "int16": "dynamic_fixed_point",
            "float16": "float16",
            "bfloat16": "bfloat16",
        }

        def get_input_for_quantize():
            for img in img_list[:cali_batch_size]:
                preprocessed_img, _, _ = self.preprocess(img)
                yield [self._np_to_tf(preprocessed_img)]

        q_net = self.nn.quantize(
            self.acuity_net,
            qtype=quantize_type,
            quantizer=q_er_table[quantize_type],
            batch_size=1,
            iterations=cali_batch_size,
            input_generator_func=get_input_for_quantize,
            compute_entropy=hybrid,
        )

        if hybrid:
            q_net = self.nn.quantize(
                q_net,
                qtype=quantize_type,
                quantizer=q_er_table[quantize_type],
                batch_size=1,
                iterations=cali_batch_size,
                input_generator_func=get_input_for_quantize,
                hybrid=True,
            )

        self.nn.save_model_quantize(q_net, f"./{self.net_name}_{quantize_type}.quantize")

    def _infer_single(self, preprocessed_img):
        """
        Run inference on a single preprocessed image.

        Returns:
            (8400, 84) ndarray
        """
        input_data = [self._np_to_tf(preprocessed_img)]
        _, output = self.nn.run_inference_session(input_data)
        return output[0].squeeze(0) # [1,300,6]
        

    def __call__(self, ori_img):
        """
        Run full pipeline on a single image.

        Args:
            ori_img: (H, W, 3) BGR ndarray

        Returns:
            (N, 6) ndarray — [x1, y1, x2, y2, score, class_id]
        """
        ori_h, ori_w = ori_img.shape[:2]
        preprocessed_img, ratio, txy = self.preprocess(ori_img)

        pred = self._infer_single(preprocessed_img)

        det = self.postprocess(pred, (ori_w, ori_h), ratio, txy)
        return det
