import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

file_bing = './bing_img.png'
file_atlas = './atlas_img.png'


def show_img(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def read_img(filename):
    img = cv2.imread(filename)
    show_img(img)

    return img


def process_img(img):
    ### greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ### gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ### canny
    edged = cv2.Canny(blurred, 50, 150)


def Segment_Robert (img):
    kernelx = np.array([[1, 0], [0, -1]])
    kernely = np.array([[0, 1], [-1, 0]])
    img_robertx = cv2.filter2D(img, -1, kernelx)
    img_roberty = cv2.filter2D(img, -1, kernely)
    grad = cv2.addWeighted(img_robertx, 0.5, img_roberty, 0.5, 0)

    plt.subplot(121), plt.imshow(img)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(grad)
    cv2.imwrite("segmentKmeans.jpg", grad)
    plt.title("robert operator"), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
    return grad


def Segment_kmeans (img):
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r, g, b])
    # --------  первинне перетворення
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    twoDimage = img.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 10
    attempts = 10
    # --------  ініціалізація методу kmeans
    ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    edged_img = res.reshape((img.shape))
    # ------- відображення результату
    plt.subplot(121), plt.imshow(rgb_img)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edged_img, 'gray')
    cv2.imwrite("segmentKmeans.jpg", edged_img)
    plt.title("kmeans operator"), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
    return edged_img


def image_recognition(image_entrance, image_cont):
    total = 0
    for c in image_cont:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            cv2.drawContours(image_entrance, [approx], -1, (0, 255, 0), 4)
            total += 1

    print("Знайдено {0} сегмент(а) прямокутних об'єктів".format(total))
    #cv2.imwrite(file_name, image_entrance)
    plt.imshow(image_entrance)
    plt.show()

    return


## Atlas image processing and building detection
img_atlas = read_img(file_atlas)

kmeans_img = Segment_kmeans(img_atlas)

gray = cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2GRAY)
show_img(gray)

blurred = cv2.GaussianBlur(gray, (5, 5), 2)
show_img(blurred)

new_img_file = "new_img.jpg"
cv2.imwrite(new_img_file, blurred)

robert_img = Segment_Robert(blurred)

edged = cv2.Canny(robert_img, 50, 250)
show_img(edged)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
show_img(closed)
#
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
#
image_recognition(closed, cnts)

## Bing image processing and building detection

#img_bing = read_img(file_bing)