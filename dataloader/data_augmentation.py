import tensorflow as tf
import numbers
import cv2

def RandomGray(img_sequence, gray_ratio = 0.25):
    def rgb_to_gray():
        gray_images = tf.image.rgb_to_grayscale(img_sequence)
        return tf.concat([gray_images] * 3, axis=3)

    def identity():
        return tf.identity(img_sequence)

    return tf.cond(tf.less(tf.random_uniform([], 0, 1), gray_ratio), rgb_to_gray, identity)

def RandomStretch(img, max_stretch=0.4, interpolation='bilinear'):
    scale = 1 + tf.random_uniform([], -max_stretch, max_stretch)
    img_shape = tf.shape(img)
    ts = tf.to_int32(tf.round(tf.to_float(img_shape[:2]) * scale))
    resize_method_map = {'bilinear': tf.image.ResizeMethod.BILINEAR,
                         'bicubic': tf.image.ResizeMethod.BICUBIC}
    return tf.image.resize_images(img, ts, method=resize_method_map[interpolation]), scale

def CenterCrop(img, size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    th, tw = size
    return tf.image.resize_image_with_crop_or_pad(img, th, tw)

def RandomCrop(img, size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    img_shape = tf.shape(img)
    (th, tw) = size
    pad_y = tf.minimum(tf.abs(tf.minimum(img_shape[0] - th - 64, 0)), img_shape[0])
    pad_x = tf.minimum(tf.abs(tf.minimum(img_shape[1] - tw - 64, 0)), img_shape[1])
    img = tf.pad(img, [[pad_y,pad_y],[pad_x,pad_x],[0,0]], constant_values = 128)
    img_shape = tf.shape(img)
    y1 = tf.random_uniform([], 0, img_shape[0] - th, tf.int32)
    x1 = tf.random_uniform([], 0, img_shape[1] - tw, tf.int32)
    return (tf.image.crop_to_bounding_box(img, y1, x1, th, tw), [x1,y1], [pad_x,pad_y])

def RandomColorAug(img, max_brightness_delta=0.12, contrast_lower=0.5, contrast_upper=1.5, prob = 0.3):
    random_aug_color = tf.random_uniform([], 0., 1., tf.float32, name = 'random_aug_color')
    color_aug = tf.image.random_brightness(img, max_brightness_delta)
    color_aug = tf.image.random_contrast(color_aug, contrast_lower, contrast_upper)
    img = tf.cond(tf.less(random_aug_color, prob), lambda: color_aug, lambda: img)
    return img

def RandomBlur(img, prob = 0.3):
    random_gaussain_blur = tf.random_uniform([], 0, 1, tf.float32, name = 'random_gaussain_blur')
    img_shape = tf.shape(img)
    def gaussian_blur(img):
        return cv2.GaussianBlur(img, (5, 5), 0)
    blur_img = tf.py_func(gaussian_blur, [img], [tf.uint8], name="blur_img")
    img = tf.cond(tf.less(random_gaussain_blur, prob), lambda: blur_img, lambda: img)
    img.set_shape([255,255,3])
    return img 

def RandomFlip(img, prob = 0.3):
    random_flip_prob_lr = tf.random_uniform([], 0, 1, tf.float32, name = 'random_flip_prob_lr')
    random_flip_prob_up = tf.random_uniform([], 0, 1, tf.float32, name = 'random_flip_prob_up')
    img = tf.cond(tf.less(random_flip_prob_lr, prob), lambda: tf.image.flip_left_right(img), lambda: img)
    img = tf.cond(tf.less(random_flip_prob_up, prob), lambda: tf.image.flip_up_down(img), lambda: img)
    return img
