import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_img(img, title):
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()



def show_img_hist(img, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img)
    axes[0].set_title(f"{title}")
    axes[1].hist(img.ravel(), 256, [0, 256])
    axes[1].set_title("Гістограма яскравості")
    plt.tight_layout()
    plt.show()


def process_kmeans(img, k):
    Z = np.float32(img.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2


def define_red(img):
    B, G, R = cv2.split(img)

    eq_B = cv2.equalizeHist(B)
    eq_G = cv2.equalizeHist(G)
    eq_R = cv2.addWeighted(R, 4, np.zeros_like(R), 0, 0.02)

    img_redder = cv2.merge((eq_B, eq_G, eq_R))

    return img_redder


def equalize_hist(img):
    B, G, R = cv2.split(img)

    eq_B = cv2.equalizeHist(B)
    eq_G = cv2.equalizeHist(G)
    eq_R = cv2.equalizeHist(R)

    img_eq = cv2.merge((eq_B, eq_G, eq_R))

    return img_eq


def mask_kpi(img):
    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)

    return mask


def s_log(img):
    c = 255 / np.log(1 + np.max(img))
    return np.array(c * (np.log(img + 1)), dtype=np.uint8)


def s_exp(img):
    c = 255 / np.log(1 + np.max(img))
    return np.array(np.exp(img**1/c)-1, dtype=np.uint8)


def gamma_correction(img, gamma):
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(img, lookUpTable)

