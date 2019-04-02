import collections
import numpy as np
import cv2 
import copy

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])


class TargetState(object):
  def __init__(self, bbox):
    self.search_box = bbox #(cx, cy, w, h) in the original image
    self.target_box = bbox
    self.scale = np.sqrt(bbox.width*bbox.height)
    self.ratio = bbox.height*1.0/bbox.width
    
def get_center(x):
  return (x - 1.) / 2.

def corrdinate_to_bbox(corrdinates):
    """
    convert corrdinates to bbox 

    corrdinates: tuple of length 4: x1,y1,x2,y2
    return bbox format: w, h, cx, cy
    """
    w = corrdinates[2] - corrdinates[0] + 1.0
    h = corrdinates[3] - corrdinates[1] + 1.0

    cx = (corrdinates[0] + corrdinates[2])/2.0
    cy = (corrdinates[1] + corrdinates[3])/2.0

    return cx,cy,w,h

def bbox_to_corrdinate(bbox):
  #Center cx,cy
  x, y, target_width, target_height = bbox.x, bbox.y, bbox.width, bbox.height
  x1 = x - (target_width-1)/2.0
  y1 = y - (target_height-1)/2.0
  x2 = x + (target_width-1)/2.0
  y2 = y + (target_height-1)/2.0

  return x1,y1,x2,y2


def im2rgb(im):
  if len(im.shape) != 3:
    im = np.stack([im, im, im], -1)
  return im


def convert_bbox_format(bbox, to):
  x, y, target_width, target_height = bbox.x, bbox.y, bbox.width, bbox.height
  if to == 'top-left-based':
    x -= get_center(target_width)
    y -= get_center(target_height)
  elif to == 'center-based':
    y += get_center(target_height)
    x += get_center(target_width)
  else:
    raise ValueError("Bbox format: {} was not recognized".format(to))
  return Rectangle(x, y, target_width, target_height)


def get_crops(im, bbox, size_z, size_x, context_amount):
  """Obtain image sub-window, padding with avg channel if area goes outside of border
  Adapted from https://github.com/bertinetto/siamese-fc/blob/master/ILSVRC15-curation/save_crops.m#L46
  Args:
    im: Image ndarray
    bbox: Named tuple (x, y, width, height) x, y corresponds to the crops center
    size_z: Target + context size
    size_x: The resultant crop size(resized)
    s_x: The crop size (unresized)
    context_amount: The amount of context

  Returns:
    image crop: Image ndarray
  """
  cy, cx, h, w = bbox.y, bbox.x, bbox.height, bbox.width
  wc_z = w + context_amount * (w + h)
  hc_z = h + context_amount * (w + h)
  s_z = np.sqrt(wc_z * hc_z)
  scale_z = size_z / s_z

  d_search = (size_x - size_z) / 2
  pad = d_search / scale_z
  s_x = s_z + 2 * pad
  scale_x = size_x / s_x

  image_crop_x, _, _, _, _ = get_subwindow_avg(im, [cy, cx],
                                               [size_x, size_x],
                                               [np.round(s_x), np.round(s_x)])
  show = False
  if show:
    x_min = int(cx-w/2.0)
    x_max = int(cx+w/2.0)
    y_min = int(cy-h/2.0)
    y_max = int(cy+h/2.0)

    cv2.rectangle(im, (x_min, y_min), (x_max, y_max), (0,0,255), 2)
    print(cx,cy, w, h)


    new_crop_w = w*1.0/s_x*size_x
    new_crop_h = h*1.0/s_x*size_x
    new_crop_x = get_center(size_x)
    new_crop_y = get_center(size_x)
    cv2.rectangle(image_crop_x, (int(new_crop_x-get_center(new_crop_w)),int(new_crop_y-get_center(new_crop_h))),
                                (int(new_crop_x+get_center(new_crop_w)),int(new_crop_y+get_center(new_crop_h))),
                                (0,0,255),2)

    cv2.imshow("img", im)
    cv2.imshow("crop",image_crop_x)
    cv2.waitKey(0)

  return image_crop_x, scale_x,[w*scale_x, h*scale_x]



def get_subwindow_avg(im, pos, model_sz, original_sz):
  # avg_chans = np.mean(im, axis=(0, 1)) # This version is 3x slower
  avg_chans = [np.mean(im[:, :, 0]), np.mean(im[:, :, 1]), np.mean(im[:, :, 2])]
  if not original_sz:
    original_sz = model_sz
  sz = original_sz
  im_sz = im.shape
  # make sure the size is not too small
  assert im_sz[0] > 2 and im_sz[1] > 2
  c = [get_center(s) for s in sz]

  # check out-of-bounds coordinates, and set them to avg_chans
  context_xmin = np.int(np.round(pos[1] - c[1]))
  context_xmax = np.int(context_xmin + sz[1] - 1)
  context_ymin = np.int(np.round(pos[0] - c[0]))
  context_ymax = np.int(context_ymin + sz[0] - 1)
  left_pad = np.int(np.maximum(0, -context_xmin))
  top_pad = np.int(np.maximum(0, -context_ymin))
  right_pad = np.int(np.maximum(0, context_xmax - im_sz[1] + 1))
  bottom_pad = np.int(np.maximum(0, context_ymax - im_sz[0] + 1))

  context_xmin = context_xmin + left_pad
  context_xmax = context_xmax + left_pad
  context_ymin = context_ymin + top_pad
  context_ymax = context_ymax + top_pad
  if top_pad > 0 or bottom_pad > 0 or left_pad > 0 or right_pad > 0:
    R = np.pad(im[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)),
               'constant', constant_values=(avg_chans[0]))
    G = np.pad(im[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)),
               'constant', constant_values=(avg_chans[1]))
    B = np.pad(im[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)),
               'constant', constant_values=(avg_chans[2]))

    im = np.stack((R, G, B), axis=2)

  im_patch_original = im[context_ymin:context_ymax + 1,
                      context_xmin:context_xmax + 1, :]
  if not (model_sz[0] == original_sz[0] and model_sz[1] == original_sz[1]):
    im_patch = cv2.resize(im_patch_original, tuple(model_sz))
  else:
    im_patch = im_patch_original
  return im_patch, left_pad, top_pad, right_pad, bottom_pad
  
def overlap(box1, box2):
    assert(np.shape(box1) == np.shape(box2))
    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)

    xA = np.maximum(x11, x21)
    yA = np.maximum(y11, y21)
    xB = np.minimum(x12, x22)
    yB = np.minimum(y12, y22)

    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB-yA+1),0)

    boxAArea = (x12 - x11+1)*(y12 - y11 +1)
    boxBArea = (x22 - x21+1)*(y22 - y21 +1)
    UnionArea = boxAArea + boxBArea - interArea
    iou = interArea*1.0/UnionArea
    return iou
def NMS(dets, scores, conf_th=0.4, overlap_th=0.1):  
    order = scores.argsort()[::-1]  
    keep_indexs = []  
    while order.size > 0:  
        i = order[0]
        if scores[i] > conf_th or len(keep_indexs)==0:
            keep_indexs.append(i)
        if order.size == 1: break
        ovr = overlap(np.tile(np.reshape(dets[i],[1,4]),(order.size-1,1)), dets[order[1:]])
        inds = np.where(ovr <= overlap_th)[0]
        order = order[inds+1]
    return keep_indexs
    
def get_topleft_bbox(region):
    region = np.array(region)
    cx = np.mean(region[0::2])
    cy = np.mean(region[1::2])
    x1 = min(region[0::2])
    x2 = max(region[0::2])
    y1 = min(region[1::2])
    y2 = max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1
    return [cx-(w-1)//2, cy-(h-1)//2, w, h]

#===================================Help function
def safe_imread(filename):
    bgr_img = cv2.imread(filename)
    while isinstance(bgr_img, type(None)):
        print("load %s failed, reload"%(filename))
        bgr_img = cv2.imread(filename)
    return bgr_img
    
def vis_square(data_raw, name="fig"):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    # normalize data for display
    data = copy.copy(data_raw)
    data = (data - data.min()) / (data.max() - data.min())
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    data = cv2.applyColorMap((255*data).astype(np.uint8), cv2.COLORMAP_JET) if np.shape(data)[-1] !=3 else data
    data = cv2.resize(data, (256,256)) if np.shape(data)[0]< 256 else data
    
    cv2.imshow(name, data)
    cv2.waitKey(1)
