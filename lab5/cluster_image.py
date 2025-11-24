import cv2
import numpy as np
import matplotlib.pyplot as plt
from img_operations import *


def remove_roads(img, total, x1, x2, y1, y2):
    roads_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary_roads = cv2.threshold(roads_blurred[y1:y2, x1:x2], 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masked_roads = np.zeros_like(total)
    masked_roads[y1:y2, x1:x2] = binary_roads
    show_img(masked_roads, "Застосування сегментації Оцу (маска доріг)")

    total = cv2.subtract(total, masked_roads)
    return total


def kpi_mask(img_eq, binary, x1, y1, x2, y2, h, w):
    mask = np.zeros((h, w), dtype="uint8")
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    masked_region = cv2.bitwise_and(img_eq, img_eq, mask=mask)
    show_img(masked_region, "Застосування маски")

    k = 10
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
    show_img(masked_res, "Застосування сегментації")

    return masked_res, log_mask_img


def find_buildings(img, k, rect_w, rect_h, x, y, gamma, min_area, red_definition=True, subtract_kpi_roads=False):
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
    show_img(binary, "Застосування сегментації Оцу")

    # обробки маски з головним корпусом КПІ
    h, w = img.shape[:2]
    x1, y1 = w - rect_w - x, y
    x2, y2 = x1 + rect_w, y1 + rect_h
    masked_res, log_mask_img = kpi_mask(img_eq, binary, x1, y1, x2, y2, h, w)

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
    buildings = [cnt for cnt in cnts if cv2.contourArea(cnt) > min_area]

    return buildings


def find_trees(img, k, gamma, min_area, thresh, red_definition=True):
    bil_filtered = cv2.bilateralFilter(img, 5, 150, 75)
    show_img(bil_filtered, "Застосування bilateral filter")

    kmeans_img = process_kmeans(bil_filtered, k)
    show_img(kmeans_img, f"Застосування сегментації kmeans ({k})")

    if red_definition:
        img_eq = define_red(kmeans_img)
        show_img_hist(img_eq, f"Нормалізація Blue, Green, виділення Red")
    else:
        img_eq = equalize_hist(kmeans_img)
        show_img_hist(img_eq, f"Нормалізація гістограми яскравості")

    gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
    show_img(gray, "Застосування градації сірого")

    corrected = gamma_correction(gray, gamma)
    show_img(corrected, "Гамма корекція")

    blurred = cv2.GaussianBlur(corrected, (9, 9), 2)
    show_img(blurred, "Застосування Гаусівського розмиття")

    inv_img = cv2.bitwise_not(blurred)
    show_img(inv_img, "Інверсія")

    _, binary = cv2.threshold(inv_img, thresh, 255, cv2.THRESH_BINARY)
    show_img(binary, "Застосування бінарної сегментації")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    show_img(morph, "Застосування морфології")

    cnts, _ = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    trees = [cnt for cnt in cnts if cv2.contourArea(cnt) > min_area]

    return trees


def process_image(img, site):
    show_img_hist(img, f"Фото з сайту {site}")
    if site == "livingatlas2":
        # for buildings
        k=6
        rect_w, rect_h = 260, 150
        x, y = 200, 70
        gamma = 0.8
        min_area = 180
        red_definition = False
        subtract_kpi_roads = False

        # for trees
        tree_k = 25
        tree_gamma = 1.8
        # 180 or 203
        tree_thresh = 180
        tree_area = 180
    else:
        # for buildings
        k = 3
        rect_w, rect_h = 300, 170
        x, y = 300, 100
        gamma = 1.5
        min_area = 210
        red_definition = True
        subtract_kpi_roads = True

        # for trees
        tree_k = 9
        tree_gamma = 1.5
        tree_thresh = 181
        tree_area = 50


    #buildings = find_buildings(img, k, rect_w, rect_h, x, y, gamma, min_area, red_definition, subtract_kpi_roads)
    trees = find_trees(img, tree_k, tree_gamma, tree_area, tree_thresh, red_definition)

    img1 = img.copy()
    img2 = img.copy()

    cv2.drawContours(img1, trees, -1, (0, 0, 255), 2)
    show_img(img1, f"Фото {site} з контурами зелених зон")

    #cv2.drawContours(img2, buildings, -1, (255, 0, 0), 2)
    #show_img(img2, f"Фото {site} з контурами будівель")

    return