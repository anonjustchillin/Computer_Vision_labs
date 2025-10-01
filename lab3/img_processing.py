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


def remove_roads(img, total, x1, x2, y1, y2):
    roads_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary_roads = cv2.threshold(roads_blurred[y1:y2, x1:x2], 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masked_roads = np.zeros_like(total)
    masked_roads[y1:y2, x1:x2] = binary_roads
    show_img(masked_roads, "Застосування сегментації Отцу (маска доріг)")

    total = cv2.subtract(total, masked_roads)
    return total



def image_recognition(cnts, min_area):
    buildings = [cnt for cnt in cnts if cv2.contourArea(cnt) > min_area]
    print(f"Знайдено {len(buildings)} будівель на фото.")
    return buildings


def process_image(img, site, k, rect_w, rect_h, x, y, gamma, min_area, red_definition=True, subtract_kpi_roads=True):
    show_img_hist(img, f"Фото з сайту {site}")

    if red_definition:
        img_eq = define_red(img)
        show_img_hist(img_eq, f"Нормалізація Blue, Green, виділення Red")
    else:
        img_eq = equalize_hist(img)
        show_img_hist(img_eq, f"Нормалізація гістограми яскравості")

    bil_filtered = cv2.bilateralFilter(img_eq, 5, 150, 75)
    show_img(bil_filtered, "Застосування bilateral filter")

    kmeans_img = process_kmeans(bil_filtered, k)
    show_img(kmeans_img, f"Застосування сегментації kmeans ({k})")

    gray = cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2GRAY)
    show_img(gray, "Застосування градації сірого")

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    show_img(blurred, "Застосування Гаусівського розмиття")

    exp_img = s_exp(blurred)
    show_img_hist(exp_img, "Застосування експоненційної трансформації")

    corrected = gamma_correction(exp_img, gamma)
    show_img(corrected, "Гамма корекція")

    _, binary = cv2.threshold(corrected, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    show_img(binary, "Застосування сегментації Отцу")

    # обробки маски з головним корпусом КПІ
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype="uint8")
    x1, y1 = w - rect_w - x, y
    x2, y2 = x1 + rect_w, y1 + rect_h
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    masked_region = cv2.bitwise_and(img_eq, img_eq, mask=mask)
    show_img(masked_region, "Застосування маски")

    k=10
    kmeans_mask_img = process_kmeans(masked_region, k)
    show_img(kmeans_mask_img, f"Застосування сегментації kmeans ({k})")

    mask_gray = cv2.cvtColor(kmeans_mask_img, cv2.COLOR_BGR2GRAY)
    show_img(mask_gray, "Застосування градації сірого")

    masked_blurred = cv2.GaussianBlur(mask_gray, (5, 5), 2)
    show_img(masked_blurred, "Застосування Гаусівського розмиття")

    log_mask_img = s_exp(masked_blurred)
    show_img(log_mask_img, "Застосування логарифмічної трансформації")

    inverted_mask = cv2.bitwise_not(log_mask_img)
    show_img(inverted_mask, "inverted mask")

    exp_mask_img = s_exp(inverted_mask)
    show_img(exp_mask_img, "Застосування експоненційної трансформації")

    gamma = 3
    corrected_mask = gamma_correction(exp_mask_img, gamma)
    show_img(corrected_mask, "Гамма корекція")

    masked_blurred2 = cv2.GaussianBlur(corrected_mask, (11, 11), 0)
    show_img(masked_blurred2, "Застосування Гаусівського розмиття")

    _, binary_masked = cv2.threshold(masked_blurred2[y1:y2, x1:x2], 220, 255, cv2.THRESH_BINARY)
    masked_res = np.zeros_like(binary)
    masked_res[y1:y2, x1:x2] = binary_masked
    show_img(masked_res, "Застосування сегментації Отцу (маска)")

    # загальне
    total = cv2.bitwise_or(binary, masked_res)
    show_img(total, "Загальна картина")

    if subtract_kpi_roads:
        total = remove_roads(log_mask_img, total, x1, x2, y1, y2)
        show_img(total, "Прибрані дороги біля головного корпусу КПІ")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(total, cv2.MORPH_CLOSE, kernel)
    show_img(morph, "Застосування морфології")

    cnts, _ = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    buildings = image_recognition(cnts, min_area)
    cv2.drawContours(img, buildings, -1, (255, 0, 0), 2)
    show_img(img, f"Фото {site} з контурами")

    return
