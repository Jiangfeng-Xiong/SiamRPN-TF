import tensorflow as tf

def tf_iou(boxes1, boxes2):
    """
        DIM(boxes1) == DIM(boxes2) == N*4
    """
    x11, y11, x12, y12 = tf.split(boxes1, 4, axis=1) #N*1
    x21, y21, x22, y22 = tf.split(boxes2, 4, axis=1) #N*1

    xA = tf.maximum(x11, x21)
    yA = tf.maximum(y11, y21)
    xB = tf.minimum(x12, x22)
    yB = tf.minimum(y12, y22)

    interArea = tf.maximum((xB - xA + 1), 0) * tf.maximum((yB-yA+1),0)

    boxAArea = (x12 - x11+1)*(y12 - y11 +1)
    boxBArea = (x22 - x21+1)*(y22 - y21 +1)
    UnionArea = boxAArea + boxBArea - interArea

    iou = tf.minimum(tf.maximum(interArea/(UnionArea+1e-10),0),1.0)
    return iou
    
def tf_giou(boxes1, boxes2):
    """
        DIM(boxes1) == DIM(boxes2) == N*4
    """
    x11, y11, x12, y12 = tf.split(boxes1, 4, axis=1) #N*1
    x21, y21, x22, y22 = tf.split(boxes2, 4, axis=1)
    
    #ensure x11<x12,y11<y12
    x11,x12,y11,y12 = tf.minimum(x11, x12), tf.maximum(x11, x12), tf.minimum(y11,y12),tf.maximum(y11,y12)
    x21,x22,y21,y22 = tf.minimum(x21, x22), tf.maximum(x21, x22), tf.minimum(y21,y22),tf.maximum(y21,y22)
    

    xA = tf.maximum(x11, x21)
    yA = tf.maximum(y11, y21)
    xB = tf.minimum(x12, x22)
    yB = tf.minimum(y12, y22)

    interArea = tf.maximum((xB - xA + 1), 0) * tf.maximum((yB-yA+1),0)

    boxAArea = (x12 - x11+1)*(y12 - y11 +1)
    boxBArea = (x22 - x21+1)*(y22 - y21 +1)
            
    unionArea = boxAArea + boxBArea - interArea

    iou = interArea/unionArea
            
    xC = tf.minimum(x11, x21)
    yC = tf.minimum(y11, y21)
    xD = tf.maximum(x12, x22)
    yD = tf.maximum(y12, y22)
    
    maxEncloseArea = tf.maximum((xD - xC + 1), 0) * tf.maximum((yD-yC+1),0)
    giou = iou - (maxEncloseArea - unionArea)/(maxEncloseArea+1e-10)

    return giou
    
def tf_bbox_transform(ex_rois, gt_rois):
    #"N*4"
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
    targets_dw = tf.log(gt_widths / ex_widths)
    targets_dh = tf.log(gt_heights / ex_heights)

    targets = tf.stack([targets_dx, targets_dy, targets_dw, targets_dh], axis=1)
    return targets

def tf_bbox_transform_inv(boxes, deltas):
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
    
    widths = tf.expand_dims(widths, axis=1)
    heights = tf.expand_dims(heights, axis=1)
    ctr_x = tf.expand_dims(ctr_x, axis=1)
    ctr_y = tf.expand_dims(ctr_y, axis=1)
   
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = tf.exp(dw) * widths
    pred_h = tf.exp(dh) * heights

    pred_boxes = tf.zeros(tf.shape(deltas), dtype=tf.float32)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes
