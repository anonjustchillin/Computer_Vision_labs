import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_img(img, title):
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()


def resize_img(img, size, padColor=0):
    h, w = img.shape[:2]
    sh, sw = size

    aspect = float(w) / h
    saspect = float(sw) / sh

    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

        # set pad color
    if len(img.shape) is 3 and not isinstance(padColor,
                                              (list, tuple, np.ndarray)):  # color image but only one color provided
        padColor = [padColor] * 3

        # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT,
                                    value=padColor)

    return scaled_img