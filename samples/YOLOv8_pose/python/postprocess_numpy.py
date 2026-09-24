
import numpy as np
import cv2

class PostProcess:
    def __init__(self, conf_thresh=0.001, nms_thresh=0.7, agnostic=False, multi_label=True, max_det=300, kpt_num=17):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.agnostic_nms = agnostic
        self.multi_label = multi_label
        self.max_det = max_det
        self.kpt_num = kpt_num


    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.copy() if isinstance(x, np.ndarray) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y


    def nms_boxes(self, pred, iou_thres):
        x = pred[:, 0]
        y = pred[:, 1]
        w = pred[:, 2] - pred[:, 0]
        h = pred[:, 3] - pred[:, 1]

        scores = pred[:, 4]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]
        output = []
        for i in keep:
            output.append(pred[i].tolist())
        return np.array(output)

    def  __call__(self, preds_batch, org_size_batch, ratios_batch, txy_batch):
        """
        post-processing
        :param preds_batch:     list of predictions in a batch, (1, n,8400,56) [cx,cy,w,h,conf,17*3]
        :param org_size_batch:  list of (org_img_w, org_img_h) in a batch
        :param ratios_batch:    list of (ratio_x, ratio_y) in a batch when resize-and-center-padding
        :param txy_batch:       list of (tx, ty) in a batch when resize-and-center-padding
        :return:
        """
        results = []
        preds = preds_batch[0]
        for i, pred in enumerate(preds):
            pred = np.transpose(pred, (1, 0))

            pred = pred[pred[:, 4] > self.conf_thresh]

            if len(pred) == 0:
                print("none detected")
                results.append(np.zeros((0, 56)))
            else:
                # Box (center x, center y, width, height) to (x1, y1, x2, y2)
                pred = self.xywh2xyxy(pred)
                results.append(self.nms_boxes(pred, self.nms_thresh))
        
        for det, (org_w, org_h), ratio, (tx1, ty1) in zip(results, org_size_batch, ratios_batch, txy_batch):
            if len(det):
                # Rescale boxes from img_size to im0 size
                coords = det[:, :4]
                coords[:, [0, 2]] -= tx1  # x padding
                coords[:, [1, 3]] -= ty1  # y padding
                coords[:, [0, 2]] /= ratio[0]
                coords[:, [1, 3]] /= ratio[1]

                coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, org_w - 1)  # x1, x2
                coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, org_h - 1)  # y1, y2

                det[:, :4] = coords

                # Rescale keypoints from img_size to im0 size
                num_kpts = (det.shape[1] - 5) // 3
                for k in range(num_kpts):
                    det[:, 5 + k * 3] -= tx1
                    det[:, 5 + k * 3 + 1] -= ty1
                    det[:, 5 + k * 3] /= ratio[0]
                    det[:, 5 + k * 3 + 1] /= ratio[1]
                    det[:, 5 + k * 3] = det[:, 5 + k * 3].clip(0, org_w - 1)
                    det[:, 5 + k * 3 + 1] = det[:, 5 + k * 3 + 1].clip(0, org_h - 1)
        
        return results

