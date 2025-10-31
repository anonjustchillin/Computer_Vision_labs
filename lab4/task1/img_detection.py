import cv2
import matplotlib.pyplot as plt
from detection import (harris_corners,
                            sift_detector,
                            shi_tomasi_detector,
                            sift_compare, harris_compare, shi_tomasi_compare)


file_bing = './bing_img.png'
file_atlas = './atlas_img.png'

def show_img(img, title):
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()

## Atlas image processing and building detection
img_atlas = cv2.imread(file_atlas)
show_img(img_atlas, "Фото з сайту livingatlas2")

atlas_harris = harris_corners(img_atlas)
show_img(atlas_harris, "Harris corners (livingatlas2)")

atlas_sift = sift_detector(img_atlas.copy())
show_img(atlas_sift, "SIFT (livingatlas2)")

atlas_shi_tomasi = shi_tomasi_detector(img_atlas.copy())
for i in atlas_shi_tomasi:
    x, y = i.ravel()
    cv2.circle(img_atlas, (x, y), 3, (255, 0, 0), -1)
show_img(img_atlas, "Shi Tomasi corners (livingatlas2)")


## Bing image processing and building detection
img_bing = cv2.imread(file_bing)
show_img(img_bing, "Фото з сайту Bing")

bing_harris = harris_corners(img_bing)
show_img(bing_harris, "Harris corners (Bing)")

bing_sift = sift_detector(img_bing.copy())
show_img(bing_sift, "SIFT (Bing)")

bing_shi_tomasi = shi_tomasi_detector(img_bing.copy())
for i in bing_shi_tomasi:
    x, y = i.ravel()
    cv2.circle(img_bing, (x, y), 3, (255, 0, 0), -1)
show_img(img_bing, "Shi Tomasi corners (Bing)")

#####################
img_atlas = cv2.imread(file_atlas)
img_bing = cv2.imread(file_bing)


def equalize_hist(img):
    B, G, R = cv2.split(img)

    eq_B = cv2.equalizeHist(B)
    eq_G = cv2.equalizeHist(G)
    eq_R = cv2.equalizeHist(R)

    img_eq = cv2.merge((eq_B, eq_G, eq_R))

    blurred = cv2.GaussianBlur(img_eq, (3, 3), 0)

    return blurred


proc_atlas = equalize_hist(img_atlas)
proc_bing = equalize_hist(img_bing)


show_img(proc_atlas, "Фото з сайту livingatlas2")
show_img(proc_bing, "Фото з сайту Bing")

harris_compare(proc_atlas, proc_bing)

sift_compare(proc_atlas, proc_bing)

shi_tomasi_compare(proc_atlas, proc_bing)

