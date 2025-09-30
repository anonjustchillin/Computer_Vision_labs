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
    B, G, R = cv2.split(img)

    eq_B = cv2.equalizeHist(B)
    eq_G = cv2.equalizeHist(G)
    eq_R = cv2.equalizeHist(R)

    img_eq = cv2.merge((eq_B, eq_G, eq_R))

    return img_eq


def s_log(img):
    c = 255 / np.log(1 + np.max(img))
    return np.array(c * (np.log(img + 1)), dtype=np.uint8)


def s_exp(img):
    c = 255 / np.log(1 + np.max(img))
    return np.array(np.exp(img**1/c)-1, dtype=np.uint8)


def image_recognition(cnts):
    min_area = 350
    buildings = [cnt for cnt in cnts if cv2.contourArea(cnt) > min_area]
    print(f"Знайдено {len(buildings)} будівель на фото.")
    return buildings


def process_image(img, site):
    show_img_hist(img, f"Фото з сайту {site}")

    B, G, R = cv2.split(img)
    eq_B = cv2.equalizeHist(B)
    eq_G = cv2.equalizeHist(G)
    eq_R = cv2.addWeighted(R, 1.5, np.zeros_like(R), 0, 0)
    img_redder = cv2.merge((eq_B, eq_G, eq_R))
    show_img_hist(img_redder, f"Більш червоний")

    # img_eq = normalize_hist(img)
    # show_img_hist(img_eq, "Фото після нормалізації гістограми")

    kmeans_img = process_kmeans(img_redder, 3)
    show_img(kmeans_img, "Застосування сегментації kmeans")

    gray = cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2GRAY)
    show_img(gray, "Застосування градації сірого")

    bil_filtered = cv2.bilateralFilter(gray, 5, 150, 75)
    show_img(bil_filtered, "Застосування bilateral filter")

    blurred = cv2.GaussianBlur(bil_filtered, (5, 5), 2)
    show_img(blurred, "Застосування Гаусівського розмиття")

    exp_img = s_exp(blurred)
    show_img_hist(exp_img, "Застосування експоненційної трансформації")

    _, binary = cv2.threshold(exp_img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    show_img(binary, "Застосування сегментації Отцу")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    show_img(morph, "Застосування морфології")

    cnts, _ = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    buildings = image_recognition(cnts)

    cv2.drawContours(img, buildings, -1, (255, 0, 0), 2)
    show_img(img, "Фото з контурами")

    return
