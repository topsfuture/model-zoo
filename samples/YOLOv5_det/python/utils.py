
import os
import sys
import numpy as np
import cv2



ANCHORS = {  # stride: anchor
    8: [[1.25000, 1.62500], [2.00000, 3.75000], [4.12500, 2.87500]],
    16: [[1.87500, 3.81250], [3.87500, 2.81250], [3.68750, 7.43750]],
    32: [[3.62500, 2.81250], [4.87500, 6.18750], [11.65625, 10.18750]]
}

COCO_CLASSES = ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COLORS = [
        [56, 0, 255],
        [226, 255, 0],
        [0, 94, 255],
        [0, 37, 255],
        [0, 255, 94],
        [255, 226, 0],
        [0, 18, 255],
        [255, 151, 0],
        [170, 0, 255],
        [0, 255, 56],
        [255, 0, 75],
        [0, 75, 255],
        [0, 255, 169],
        [255, 0, 207],
        [75, 255, 0],
        [207, 0, 255],
        [37, 0, 255],
        [0, 207, 255],
        [94, 0, 255],
        [0, 255, 113],
        [255, 18, 0],
        [255, 0, 56],
        [18, 0, 255],
        [0, 255, 226],
        [170, 255, 0],
        [255, 0, 245],
        [151, 255, 0],
        [132, 255, 0],
        [75, 0, 255],
        [151, 0, 255],
        [0, 151, 255],
        [132, 0, 255],
        [0, 255, 245],
        [255, 132, 0],
        [226, 0, 255],
        [255, 37, 0],
        [207, 255, 0],
        [0, 255, 207],
        [94, 255, 0],
        [0, 226, 255],
        [56, 255, 0],
        [255, 94, 0],
        [255, 113, 0],
        [0, 132, 255],
        [255, 0, 132],
        [255, 170, 0],
        [255, 0, 188],
        [113, 255, 0],
        [245, 0, 255],
        [113, 0, 255],
        [255, 188, 0],
        [0, 113, 255],
        [255, 0, 0],
        [0, 56, 255],
        [255, 0, 113],
        [0, 255, 188],
        [255, 0, 94],
        [255, 0, 18],
        [18, 255, 0],
        [0, 255, 132],
        [0, 188, 255],
        [0, 245, 255],
        [0, 169, 255],
        [37, 255, 0],
        [255, 0, 151],
        [188, 0, 255],
        [0, 255, 37],
        [0, 255, 0],
        [255, 0, 170],
        [255, 0, 37],
        [255, 75, 0],
        [0, 0, 255],
        [255, 207, 0],
        [255, 0, 226],
        [255, 245, 0],
        [188, 255, 0],
        [0, 255, 18],
        [0, 255, 75],
        [0, 255, 151],
        [255, 56, 0],
        [245, 255, 0],
    ]

def is_img(file_name):
    """judge the file is available image or not
    Args:
        file_name (str): input file name
    Returns:
        (bool) : whether the file is available image
    """
    fmt = os.path.splitext(file_name)[-1]
    if isinstance(fmt, str) and fmt.lower() in ['.jpg','.png','.jpeg','.bmp','.jpeg','.webp']:
        return True
    else:
        return False

# scipy: https://zhuanlan.zhihu.com/p/375344904
def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """Compute the log of the sum of exponentials of input elements.
    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.
        .. versionadded:: 0.11.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.
        .. versionadded:: 0.15.0
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`. These values may be negative in order to
        implement subtraction.
        .. versionadded:: 0.12.0
    return_sign : bool, optional
        If this is set to True, the result will be a pair containing sign
        information; if False, results that are negative will be returned
        as NaN. Default is False (no sign information).
        .. versionadded:: 0.16.0
    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.
    sgn : ndarray
        If return_sign is True, this will be an array of floating-point
        numbers matching res and +1, 0, or -1 depending on the sign
        of the result. If False, only one result is returned.

    """
    # a = _asarray_validated(a, check_finite=False)
    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -np.inf

    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out

# scipy.special.softmax: https://zhuanlan.zhihu.com/p/375344904
def softmax(x, axis=None):
    r"""
    Softmax function
    The softmax function transforms each element of a collection by
    computing the exponential of each element divided by the sum of the
    exponentials of all the elements. That is, if `x` is a one-dimensional
    numpy array::
        softmax(x) = np.exp(x)/sum(np.exp(x))
    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None and softmax will be
        computed over the entire array `x`.
    Returns
    -------
    s : ndarray
        An array the same shape as `x`. The result will sum to 1 along the
        specified axis.

    """
    # compute in log space for numerical stability
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))



 
def nms(boxes, scores, iou_thres):
    """Single class NMS implemented in Numpy."""    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
 
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]   #scores 倒序
 
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
 
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1 + 0.00001)    #pseudo_torch_nms ，代替torchvision.ops.nms
        h = np.maximum(0.0, yy2 - yy1 + 0.00001)      
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)          # iou计算
 
        inds = np.where(ovr <= iou_thres)[0]          
        order = order[inds + 1]

    keep = np.array(keep)
    return keep






def multiclass_nms_class_aware(boxes, scores, iou_thres, score_thres):   
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    """
    输入boxes.shape=(25200, 4); scores.shape=(25200, 1)
    输出final_dets ,shape（x, 6）, 通过append dets得到的一个数组，dets 是轴1拼接的三个2维数组：valid_boxes( x, 4),valid_scores(x, 1),cls_inds(x, 1); x  = len (keep)
    """
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):              # 遍历80class
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thres        # 筛一， score_thres
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, iou_thres)    #筛二， iou_thers nms。
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)               #dets 返回的是？
 
def multiclass_nms_class_agnostic(boxes, scores, iou_thres, score_thres): # scores.shape(25200,80)
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""      
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
 
    valid_score_mask = cls_scores > score_thres
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, iou_thres)
    if keep is not None:  
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1)
    return dets
 
def multiclass_nms(boxes, scores, iou_thres, conf_thres, class_agnostic=False):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, iou_thres, conf_thres)

def make_grid(nx, ny, stride, anchor):
    stride = np.array(stride)
    anchor = np.array(anchor)
    # 创建了一个坐标网格
    xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
    grid = np.stack((xv, yv), -1)
    anchor_grid = (anchor * stride).reshape(1, len(anchor), 1, 1, 2)
    return grid, anchor_grid

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))