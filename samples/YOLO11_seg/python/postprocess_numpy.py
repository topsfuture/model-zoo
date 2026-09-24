import numpy as np
import cv2


def xywh2xyxy(x):
    """Convert xywh to xyxy."""
    y = x.copy()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


class NMS:
    """Pure numpy NMS."""

    @staticmethod
    def nms(boxes, scores, iou_thres):
        """boxes: (N, 4) xyxy, scores: (N,)"""
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            inter_w = np.maximum(0.0, xx2 - xx1 + 1e-5)
            inter_h = np.maximum(0.0, yy2 - yy1 + 1e-5)
            inter = inter_w * inter_h
            area_i = (boxes[i, 2] - boxes[i, 0] + 1e-5) * (boxes[i, 3] - boxes[i, 1] + 1e-5)
            areas = (boxes[order[1:], 2] - boxes[order[1:], 0] + 1e-5) * \
                    (boxes[order[1:], 3] - boxes[order[1:], 1] + 1e-5)
            ovr = inter / (area_i + areas - inter)
            order = order[np.where(ovr <= iou_thres)[0] + 1]
        return np.array(keep, dtype=np.int64)


class PostProcess:
    """YOLO11 segmentation postprocess — aligned with C++ implementation."""

    def __init__(self, conf_thresh=0.25, nms_thresh=0.45, max_det=300,
                 mask_thresh=0.5, protos_shape=(32, 160, 160)):
        self.conf_thresh = float(conf_thresh)
        self.nms_thresh = float(nms_thresh)
        self.max_det = int(max_det)
        self.mask_thresh = float(mask_thresh)
        self.protos_h = int(protos_shape[1])
        self.protos_w = int(protos_shape[2])
        self._nms = NMS()

    def __call__(self, preds, protos, letterbox_size, org_size):
        """
        Args:
            preds: (1, 116, 8400) detection output
            protos: (1, 32, 160, 160) prototype masks
            letterbox_size: (lb_w, lb_h) = (640, 640)
            org_size: (org_w, org_h)
        Returns:
            dict: boxes(N,4), scores(N,), class_ids(N,), masks(H,W,N)
        """
        org_w, org_h = org_size
        lb_w, lb_h = letterbox_size

        # Compute letterbox params (matches C++ preprocess_image)
        ratio = min(lb_h / org_h, lb_w / org_w)
        new_w = int(org_w * ratio)
        new_h = int(org_h * ratio)
        pad_left = (lb_w - new_w) // 2
        pad_top = (lb_h - new_h) // 2

        # Run NMS in letterbox space
        det = self._non_max_suppression(preds)

        if len(det) == 0:
            return {
                'boxes': np.zeros((0, 4), dtype=np.float32),
                'scores': np.zeros((0,), dtype=np.float32),
                'class_ids': np.zeros((0,), dtype=np.int32),
                'masks': np.zeros((org_h, org_w, 0), dtype=np.bool_),
            }

        # det: (N, 6+32) = xyxy(letterbox), score, class_id, mask_coef(32)
        boxes_lb = det[:, :4].copy()
        scores = det[:, 4]
        class_ids = det[:, 5].astype(np.int32)
        mask_coefs = det[:, 6:]  # (N, 32) raw

        # Generate masks in letterbox space (matches C++ generate_masks)
        protos_3d = protos[0]  # (32, 160, 160)
        masks_lb = self._generate_masks_cpp_style(
            protos_3d, mask_coefs, boxes_lb, lb_h, lb_w
        )

        # Inverse transform boxes and masks to original image space
        # (matches C++ inverse_coordinates_and_mask)
        boxes_org = np.zeros_like(boxes_lb)
        boxes_org[:, [0, 2]] = (boxes_lb[:, [0, 2]] - pad_left) / ratio
        boxes_org[:, [1, 3]] = (boxes_lb[:, [1, 3]] - pad_top) / ratio
        boxes_org[:, [0, 2]] = boxes_org[:, [0, 2]].clip(0, org_w - 1)
        boxes_org[:, [1, 3]] = boxes_org[:, [1, 3]].clip(0, org_h - 1)

        # Inverse transform masks
        masks_org = self._inverse_mask(
            masks_lb, org_h, org_w, ratio, pad_left, pad_top
        )

        return {
            'boxes': boxes_org,
            'scores': scores,
            'class_ids': class_ids,
            'masks': masks_org,
        }

    def _generate_masks_cpp_style(self, protos, mask_coefs, boxes_lb, lb_h, lb_w):
        """
        Generate masks matching C++ generate_masks() exactly.

        protos: (32, 160, 160) CHW
        mask_coefs: (N, 32) raw logits
        boxes_lb: (N, 4) xyxy in letterbox space
        """
        N = mask_coefs.shape[0]
        if N == 0:
            return np.zeros((lb_h, lb_w, 0), dtype=np.bool_)

        # C++ code: mask = sigmoid(protos @ mask_coef)
        # protos: (32, 160, 160) -> (32, 25600)
        protos_flat = protos.reshape(32, -1)  # (32, 25600)
        # mask_coefs: (N, 32) raw (no sigmoid before matmul!)
        mask_logits = mask_coefs @ protos_flat  # (N, 25600)
        mask_maps = 1.0 / (1.0 + np.exp(-mask_logits))  # sigmoid
        mask_maps = mask_maps.reshape(N, self.protos_h, self.protos_w)  # (N, 160, 160)

        # Create full-size masks in letterbox space
        masks_lb = np.zeros((lb_h, lb_w, N), dtype=np.float32)

        scale = self.protos_w / float(lb_w)  # 160/640 = 0.25

        for i in range(N):
            # Bounding box in letterbox space (clipped)
            x1 = max(0.0, min(float(boxes_lb[i, 0]), float(lb_w)))
            y1 = max(0.0, min(float(boxes_lb[i, 1]), float(lb_h)))
            x2 = max(0.0, min(float(boxes_lb[i, 2]), float(lb_w)))
            y2 = max(0.0, min(float(boxes_lb[i, 3]), float(lb_h)))

            box_w = x2 - x1
            box_h = y2 - y1
            if box_w <= 0 or box_h <= 0:
                continue

            # Crop region in protos space
            p_x1 = int(x1 * scale)
            p_y1 = int(y1 * scale)
            p_x2 = int(x2 * scale)
            p_y2 = int(y2 * scale)

            # Ensure valid crop region
            p_x2 = max(p_x1 + 1, min(p_x2, self.protos_w))
            p_y2 = max(p_y1 + 1, min(p_y2, self.protos_h))

            # Crop and resize (matches C++ cv::resize with INTER_LINEAR)
            mask_crop = mask_maps[i, p_y1:p_y2, p_x1:p_x2]
            box_w_int = int(box_w)
            box_h_int = int(box_h)
            mask_resized = cv2.resize(mask_crop, (box_w_int, box_h_int),
                                       interpolation=cv2.INTER_LINEAR)

            # Paste to letterbox space (binarize here)
            # Handle boundary: simulate cv::Rect intersection
            x1_paste = max(0, int(x1))
            y1_paste = max(0, int(y1))
            x2_paste = min(lb_w, int(x1) + box_w_int)
            y2_paste = min(lb_h, int(y1) + box_h_int)

            src_x1 = x1_paste - int(x1)
            src_y1 = y1_paste - int(y1)
            src_x2 = src_x1 + (x2_paste - x1_paste)
            src_y2 = src_y1 + (y2_paste - y1_paste)

            masks_lb[y1_paste:y2_paste, x1_paste:x2_paste, i] = \
                mask_resized[src_y1:src_y2, src_x1:src_x2]

        return masks_lb

    def _inverse_mask(self, masks_lb, org_h, org_w, ratio, pad_left, pad_top):
        """Inverse transform masks from letterbox to original image space."""
        N = masks_lb.shape[2]
        if N == 0:
            return np.zeros((org_h, org_w, 0), dtype=np.bool_)

        lb_h, lb_w = masks_lb.shape[:2]

        # Unpad region in letterbox space
        new_w = int(org_w * ratio)
        new_h = int(org_h * ratio)
        unpad_x1 = pad_left
        unpad_y1 = pad_top
        unpad_x2 = pad_left + new_w
        unpad_y2 = pad_top + new_h

        # Crop unpadded region
        unpadded = masks_lb[unpad_y1:unpad_y2, unpad_x1:unpad_x2, :]  # (H_unpad, W_unpad, N)

        # Resize to original image size
        masks_org = np.zeros((org_h, org_w, N), dtype=np.bool_)
        for i in range(N):
            mask_2d = unpadded[:, :, i]
            mask_resized = cv2.resize(mask_2d, (org_w, org_h),
                                       interpolation=cv2.INTER_LINEAR)
            # Binarize (matches C++ threshold)
            masks_org[:, :, i] = (mask_resized > self.mask_thresh)

        return masks_org

    def _non_max_suppression(self, preds):
        """
        NMS matching C++ generate_proposals.

        ONNX layout: (1, 116, 8400)
        Channel: 0:4=xywh, 4:84=class_logits(80), 84:116=mask_coef(32)

        Key: Filter by raw class logits (no sigmoid), output score = raw max logit
        """
        # (1, 116, 8400) -> (116, 8400) -> (8400, 116)
        pred = preds[0].T  # (8400, 116)

        # xywh -> xyxy
        box_xyxy = xywh2xyxy(pred[:, 0:4])

        # Extract raw components
        cls_raw = pred[:, 4:84]      # (8400, 80) class logits
        mask_raw = pred[:, 84:116]   # (8400, 32) mask coefficients

        # Filter by raw class max > conf_thresh (matches C++: class_score < prob_threshold)
        cls_max = cls_raw.max(axis=1)  # (8400,)
        keep = cls_max > self.conf_thresh

        box_filt = box_xyxy[keep]
        cls_filt = cls_raw[keep]
        mask_filt = mask_raw[keep]

        N = box_filt.shape[0]
        if N == 0:
            return np.zeros((0, 6 + 32))

        # Get best class (score = raw logit, not sigmoid)
        scores = cls_filt.max(axis=1)  # (N,)
        class_ids = cls_filt.argmax(axis=1).astype(np.float32)  # (N,)

        # Sort by score descending
        order = scores.argsort()[::-1]
        box_sorted = box_filt[order]
        scores_sorted = scores[order]
        class_sorted = class_ids[order]
        mask_sorted = mask_filt[order]

        # Class-wise NMS (offset by class)
        offset = class_sorted * 7680  # 7680 > max box coord
        nms_boxes = box_sorted + offset[:, None]

        keep_idx = self._nms.nms(nms_boxes, scores_sorted, self.nms_thresh)
        if len(keep_idx) > self.max_det:
            keep_idx = keep_idx[:self.max_det]

        # Output: xyxy, score, class_id, mask_coef(32)
        out = np.concatenate([
            box_sorted[keep_idx],
            scores_sorted[keep_idx, None],
            class_sorted[keep_idx, None],
            mask_sorted[keep_idx],
        ], axis=1)

        return out
