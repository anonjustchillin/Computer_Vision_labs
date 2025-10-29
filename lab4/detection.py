import cv2
import numpy as np
import matplotlib.pyplot as plt
from lab4.img_processing import process_image

file_bing = './bing_img.png'
file_atlas = './atlas_img.png'


def show_img(img, title):
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()


def harris_corners(img, site):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    float_img = np.float32(img_grey)
    dest = cv2.cornerHarris(float_img, 8, 15, 0.05)
    dest = cv2.dilate(dest, None)

    img[dest > 0.01 * dest.max()] = [0, 0, 255]
    img_rgb = cv2.cvtColor(img_atlas, cv2.COLOR_BGR2RGB)

    show_img(img_rgb, f'Harris Corners ({site} image)')
    return dest


def sift_detector(img, file):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp = sift.detect(img_grey, None)

    kp_img = cv2.drawKeypoints(img_grey,
                            kp,
                            img)
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

    # cv2.imwrite(file, kp_img)
    show_img(kp_img, 'SIFT')
    return kp_img


def shi_tomasi_detector(img, site):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(img_grey, 25, 0.01, 10)
    corners = np.intp(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, (0,0,255), -1)

    show_img(img, f'Shi Tomasi corners ({site} image)')



## Atlas image processing and building detection
img_atlas = cv2.imread(file_atlas)
#show_img(img_atlas, "Фото з сайту livingatlas2")
processed_atlas = process_image(img_atlas, "livingatlas2",
                                    6, 260, 150,
                                    200, 70, 0.8, False)
#show_img(processed_atlas, "Оброблене фото livingatlas2")
#img_atlas_harris = harris_corners(img_atlas, "livingatlas2")
#img_atlas_sift = sift_detector(img_atlas, file_atlas)
img_atlas_st = shi_tomasi_detector(img_atlas, "livingatlas2")


## Bing image processing and building detection
img_bing = cv2.imread(file_bing)
#show_img(img_bing, "Фото з сайту Bing")
processed_bing = process_image(img_bing, "Bing",
                                   3, 300, 170,
                                   300, 100, 1.5)
#show_img(processed_bing, "Оброблене фото Bing")
#img_bing_harris = harris_corners(img_bing, "Bing")
#img_bing_sift = sift_detector(img_bing, file_bing)
img_bing_st = shi_tomasi_detector(img_bing, "Bing")

