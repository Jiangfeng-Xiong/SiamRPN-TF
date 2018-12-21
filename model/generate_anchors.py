"""
generate anchor boxes
"""
import numpy as np

import os,sys
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

from utils.bbox_ops_utils import iou as np_iou
from utils.bbox_transform_utils import bbox_transform as bbox_transform


anchor_config={'POS_ANCHOR_THRESHOLD': 0.6, 
                   'NEG_ANCHOR_THRESHOLD': 0.3, 
                   'MAX_POS_NUM': 16,
                   'BATCH_PER_IMAGE': 64}

def generate_anchor_topleft(base_size=16, ratios=[0.33, 0.5, 1.0, 2.0, 3.0], scales=[64,128]):
    """
    generate anchor boxes with topleft point, toal anchor nums in topleft: len(ratios)*len(scales)
    anchor boxes format: (topleft_x, topleft_y, bottomright_x, bottomright_y)

    base_size: one pixel in the feature map equals base_size pixels in the input image, i.e. the stride
    ratios: the ratios of anchor boxes
	scales: (the size of the ancho in the input image)/(feature_map_size)
	"""

    base_anchor = np.array([0, 0, base_size-1, base_size-1]) #in the input image corrdinate

    #handle different ratio anchor
    w,h,cx,cy = _anchor_to_bbox(base_anchor)
    base_anchor_area = w*h
  
    new_w = np.sqrt(base_anchor_area/ratios)
    new_h = new_w * ratios
    ratio_anchors = _make_anchors(new_w, new_h, cx, cy)

    #handle different scales
    anchors = np.zeros(shape=[len(ratios)*len(scales),4])
    for index, scale in enumerate(scales):
        ws = new_w * scale
        hs = new_h * scale
        anchors[index*len(ratios): (index+1)*len(ratios)] = _make_anchors(ws, hs, cx, cy)

    return anchors

def generate_anchor_all(base_size=None, anchor_sizes=None, ratios=None, field_size=17, net_shift=63):
    """
    shift the topleft anchors, and generate all the anchor in this feature map
    net_shift: (instance_size - 1)/2 - (field_size-1)/2 * base_size
    """    
    assert base_size!=None and anchor_sizes!=None and ratios!=None, "anchor config not set"
    topleft_anchors = generate_anchor_topleft(base_size = base_size, ratios=ratios,
                                              scales = np.array(anchor_sizes)/base_size)
    topleft_anchors = topleft_anchors + net_shift

    shifts = np.arange(0, field_size) * base_size
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.flatten()
    shift_y = shift_y.flatten()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    K = shifts.shape[0]
    A = topleft_anchors.shape[0]
    field_of_anchors = (topleft_anchors.reshape((1, A, 4)) +
                        shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    field_of_anchors = field_of_anchors.reshape((field_size, field_size, A, 4))
    field_of_anchors = field_of_anchors.astype('float32')
    field_of_anchors[:, :, :, [2, 3]] += 1

    return field_of_anchors

def get_rpn_label(anchors, gt_bboxes):
    """
    Label each anchor as +1(pos), -1(ignore), 0 
    anchor: Na(=N*N*K)*4, N*N is feature map size, K=len(scales)*len(ratios)
    gt_bboxes: Ng*4 (for SiameseRPN, we have unique bounding box, i.e. Ng=1)
    anchor_config: 
    example
        anchor_config={'POS_ANCHOR_THRESHOLD': 0.6, 
                       'NEG_ANCHOR_THRESHOLD': 0.3, 
                       'MAX_POS_NUM': 16,
                       'BATCH_PER_IMAGE': 64}
    """
    def filter_box_label(labels, value, max_example_num):
        """filter out max_exmaple_num in the original anchors, set pos/neg anchor to be ignored(-1)
        """
        curr_inds = np.where(labels == value)[0]
        if len(curr_inds) > max_example_num:
            drop_inds = np.random.choice(curr_inds, size=(len(curr_inds)-max_example_num), replace=False)
            labels[drop_inds] = -1
            curr_inds = np.where(labels == value)[0]
        return labels, curr_inds
    Na, Ng = len(anchors), len(gt_bboxes)
    assert Na>0 and Ng>0

    box_iou = np_iou(anchors, gt_bboxes) #box_iou: Na*Ng
    iou_argmax_per_anchor = np.argmax(box_iou, axis=1) # find the most overlap gt_box index for each anchor
    iou_max_per_anchor = np.array([box_iou[i,iou_argmax_per_anchor[i]] for i in range(box_iou.shape[0])])
    
    #print(iou_max_per_anchor)
    iou_max_per_gt = np.max(box_iou, axis=0, keepdims=True)
    anchor_with_max_overlap_index = np.where(box_iou == iou_max_per_gt)[0]

    anchor_labels = -1*np.ones(shape=Na,dtype=np.int32)
    
    anchor_labels[iou_max_per_anchor > anchor_config['POS_ANCHOR_THRESHOLD']] = 1
    anchor_labels[iou_max_per_anchor < anchor_config['NEG_ANCHOR_THRESHOLD']] = 0
    anchor_labels[anchor_with_max_overlap_index] = 1
   
    #filter anchors
    #pos anchors
    anchor_labels, pos_inds = filter_box_label(anchor_labels, 1, anchor_config['MAX_POS_NUM'])
    anchor_labels, neg_inds = filter_box_label(anchor_labels, 0, anchor_config['BATCH_PER_IMAGE'] - len(pos_inds))
    
    #set anchor boxes here, for negative anchor and ignored anchor, we don't need the location of the anchor since
    #they do not contribute to the regression loss, only anchor label is enough for them
    anchor_boxes = np.zeros((Na,4),dtype=np.float32)
    pos_boxes = gt_bboxes[iou_argmax_per_anchor[pos_inds],:]
    anchor_boxes[pos_inds,:] = pos_boxes

    target_boxes = bbox_transform(anchors, anchor_boxes)

    #print("%s in line %d: finally get %d pos anchors, %d neg anchors for this image"%(__file__, sys._getframe().f_lineno,
    #                        len(pos_inds),len(neg_inds))) 
    return target_boxes, anchor_labels

###############help function######################
def _anchor_to_bbox(anchor):
    """
    convert anchor to bbox 

    anchor: tuple of length 4: x1,y1,x2,y2
    return bbox format: w, h, cx, cy
    """
    w = anchor[2] - anchor[0] + 1.0
    h = anchor[3] - anchor[1] + 1.0

    cx = (anchor[0] + anchor[2])/2.0
    cy = (anchor[1] + anchor[3])/2.0

    return w, h, cx, cy

def _make_anchors(widths, heights, cx, cy):
    """
    generate anchors around centor point (cx, cy) with a list of widths and heights

    """
    anchors = np.stack((cx - (widths-1)/2.0,
                        cy - (heights-1)/2.0,
                        cx + (widths-1)/2.0,
                        cy + (heights-1)/2.0),
                        axis=1)
    return anchors

def _filter_boxes_inside(boxes, shape):
    assert boxes.ndim == 2, boxes.shape
    assert len(shape) == 2, shape

    h,w = shape

    indices = np.where(
         boxes[:,0] >=0 &
         boxes[:,1] >=0 &
         boxes[:,2] <=w &
         boxes[:,3] <=h)[0]
    return indices, boxes[indices,:]


if __name__ == "__main__":
    import cv2
    anchors = generate_anchor_all(base_size=8, ratios=[0.33,0.5,1.0,2.0,3.0], 
                                  anchor_sizes=[64],field_size=17, net_shift=63)
    img = np.zeros(shape=[255,255,3])
    test_anchor_img = img.copy()
    test_pos_anchor_img = img.copy()
    test_neg_anchor_img = img.copy()
    #print(anchors.shape)
    anchors = np.reshape(anchors, [-1, 4])
    for index, anchor in enumerate(anchors):
        #print(anchor)
        x1,y1,x2,y2 = anchor[0], anchor[1], anchor[2], anchor[3]
        w,h,cx, cy = _anchor_to_bbox(anchor)
        if abs(cx-127)<8 and abs(cy-127)<8:
            cv2.rectangle(test_anchor_img, (x1,y1),(x2,y2),list(np.random.choice(range(256), size=3)))

    cv2.imwrite("test_anchor.png", test_anchor_img)
    anchor_config={'POS_ANCHOR_THRESHOLD': 0.6, 
                   'NEG_ANCHOR_THRESHOLD': 0.3, 
                   'MAX_POS_NUM': 16,
                   'BATCH_PER_IMAGE': 64}
    anchor_boxes, labels = get_rpn_label(anchors, gt_bboxes=np.array([[127-32,127-32,127+32,127+32]]))
    #print(labels.shape)

    for i in range(labels.shape[0]):
        x1,y1,x2,y2 = anchors[i][0], anchors[i][1], anchors[i][2], anchors[i][3]
        if labels[i]==1:
            cv2.rectangle(test_pos_anchor_img , (x1,y1),(x2,y2),list(np.random.choice(range(256), size=3)))
        elif labels[i]==0:
            cv2.rectangle(test_neg_anchor_img , (x1,y1),(x2,y2),list(np.random.choice(range(256), size=3)))
    cv2.imwrite("test_pos_anchor.png",test_pos_anchor_img)
    cv2.imwrite("test_neg_anchor.png",test_neg_anchor_img)
