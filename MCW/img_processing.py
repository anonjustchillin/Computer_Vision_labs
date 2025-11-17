import cv2
import numpy as np
from img_operations import *


def equalize_hist(img):
    B, G, R = cv2.split(img)

    eq_B = cv2.equalizeHist(B)
    eq_G = cv2.equalizeHist(G)
    eq_R = cv2.equalizeHist(R)

    img_eq = cv2.merge((eq_B, eq_G, eq_R))

    return img_eq


def process_img(img):
    # нормалізація гістограми яскравості
    img = equalize_hist(img)

    # перевід в greyscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.bilateralFilter(img, 7, 15, 15)

    # налаштування різкості
    blurred = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    return img