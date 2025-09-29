import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def normalize_hist(img):
    R, G, B = cv2.split(img)

    eq_R = cv2.equalizeHist(R)
    eq_G = cv2.equalizeHist(G)
    eq_B = cv2.equalizeHist(B)

    img_eq = cv2.merge((eq_R, eq_G, eq_B))

    return img_eq


def image_recognition(cnts):
    min_area = 300
    buildings = [cnt for cnt in cnts if cv2.contourArea(cnt) > min_area]
    print(f"Знайдено {len(buildings)} будівель на фото.")
    return buildings


def process_image(img, site, k, thr):
    show_img_hist(img, f"Фото з сайту {site}")

    img_eq = normalize_hist(img)
    show_img_hist(img_eq, "Фото після нормалізації гістограми")

    kmeans_img = process_kmeans(img_eq, k)
    show_img(kmeans_img, f"Застосування сегментації kmeans (k={k})")

    gray = cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2GRAY)
    show_img(gray, "Застосування градації сірого")

    blurred = cv2.GaussianBlur(gray, (5, 5), 3)
    show_img(blurred, "Застосування Гаусівського розмиття")

    _, thresh_img = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY)
    show_img(thresh_img, "Застосування сегментації Threshold")

    cnts, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    buildings = image_recognition(cnts)

    cv2.drawContours(img, buildings, -1, (255, 0, 0), 2)
    show_img(img, "Фото з контурами")

    return
