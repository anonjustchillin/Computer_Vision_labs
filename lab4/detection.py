import cv2
import numpy as np


def harris_corners(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    float_img = np.float32(img_grey)
    dest = cv2.cornerHarris(float_img, 2, 3, 0.04)
    dest = cv2.dilate(dest, None)

    img[dest > 0.01 * dest.max()] = [0, 0, 255]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb


def sift_detector(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp = sift.detect(img_grey, None)

    kp_img = cv2.drawKeypoints(img_grey,
                            kp,
                            img)
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

    return kp_img


def shi_tomasi_detector(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(img_grey, 1200, 0.01, 20)
    corners = np.intp(corners)

    return corners


def flann(des1, des2):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    return matchesMask, matches



def sift_compare(img1, img2):
    grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(grey1, None)
    kp2, des2 = sift.detectAndCompute(grey2, None)

    matchesMask, matches = flann(des1, des2)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    final_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    cv2.imshow('sift_feature_matching', final_img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    return final_img


def harris_compare(img1, img2):
    grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    float1 = np.float32(grey1)
    float2 = np.float32(grey2)

    dest1 = cv2.cornerHarris(float1, 2, 3, 0.01)
    dest1 = cv2.dilate(dest1, None)
    ret1, dst_thresh1 = cv2.threshold(dest1, 0.01 * dest1.max(), 255, 0)
    dst_thresh1 = np.uint8(dst_thresh1)

    dest2 = cv2.cornerHarris(float2, 2, 3, 0.04)
    dest2 = cv2.dilate(dest2, None)
    ret2, dst_thresh2 = cv2.threshold(dest2, 0.01 * dest2.max(), 255, 0)
    dst_thresh2 = np.uint8(dst_thresh2)

    ret1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(dst_thresh1)
    ret2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(dst_thresh2)

    kps1 = [cv2.KeyPoint(x=float(c[0]), y=float(c[1]), size=3) for c in centroids1]
    kps2 = [cv2.KeyPoint(x=float(c[0]), y=float(c[1]), size=3) for c in centroids2]

    sift = cv2.SIFT_create()
    kp1, des1 = sift.compute(grey1, kps1)
    kp2, des2 = sift.compute(grey2, kps2)

    matchesMask, matches = flann(des1, des2)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    final_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    cv2.imshow('harris_feature_matching', final_img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    return final_img


def shi_tomasi_compare(img1, img2):
    grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    corners1 = cv2.goodFeaturesToTrack(grey1, 1200, 0.01, 20)
    corners1 = np.intp(corners1)

    corners2 = cv2.goodFeaturesToTrack(grey2, 1200, 0.01, 20)
    corners2 = np.intp(corners2)

    kps1 = [cv2.KeyPoint(x=float(x), y=float(y), size=3) for [[x, y]] in corners1]
    kps2 = [cv2.KeyPoint(x=float(x), y=float(y), size=3) for [[x, y]] in corners2]

    sift = cv2.SIFT_create()
    kp1, des1 = sift.compute(grey1, kps1)
    kp2, des2 = sift.compute(grey2, kps2)

    matchesMask, matches = flann(des1, des2)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    final_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    cv2.imshow('shi_tomasi_feature_matching', final_img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    return final_img

