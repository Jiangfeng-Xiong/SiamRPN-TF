import numpy as np


def area(boxes):
  """Computes area of boxes.

  Args:
    boxes: Numpy array with shape [N, 4] holding N boxes

  Returns:
    a numpy array with shape [N*1] representing box areas
  """
  return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2):
  """Compute pairwise intersection areas between boxes.

  Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes
    boxes2: a numpy array with shape [M, 4] holding M boxes

  Returns:
    a numpy array with shape [N*M] representing pairwise intersection area
  """
  [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
  [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

  all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
  all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
  intersect_heights = np.maximum(
      np.zeros(all_pairs_max_ymin.shape, dtype='f4'),
      all_pairs_min_ymax - all_pairs_max_ymin)
  all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
  all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
  intersect_widths = np.maximum(
      np.zeros(all_pairs_max_xmin.shape, dtype='f4'),
      all_pairs_min_xmax - all_pairs_max_xmin)
  return intersect_heights * intersect_widths


def np_iou(boxes1, boxes2):
  """Computes pairwise intersection-over-union between box collections.

  Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes.
    boxes2: a numpy array with shape [M, 4] holding M boxes.

  Returns:
    a numpy array with shape [N, M] representing pairwise iou scores.
  """
  intersect = intersection(boxes1, boxes2)
  area1 = area(boxes1)
  area2 = area(boxes2)
  union = np.expand_dims(area1, axis=1) + np.expand_dims(
      area2, axis=0) - intersect
  return intersect / union


def ioa(boxes1, boxes2):
  """Computes pairwise intersection-over-area between box collections.

  Intersection-over-area (ioa) between two boxes box1 and box2 is defined as
  their intersection area over box2's area. Note that ioa is not symmetric,
  that is, IOA(box1, box2) != IOA(box2, box1).

  Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes.
    boxes2: a numpy array with shape [M, 4] holding N boxes.

  Returns:
    a numpy array with shape [N, M] representing pairwise ioa scores.
  """
  intersect = intersection(boxes1, boxes2)
  inv_areas = np.expand_dims(1.0 / area(boxes2), axis=0)
  return intersect * inv_areas

def np_bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def np_bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes