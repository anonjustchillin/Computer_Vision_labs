import cv2
import numpy as np
import matplotlib.pyplot as plt

# harris
blockSize = 4
kSize = 3
k = 0.03

# shi tomasi
maxCorners = 400
qualityLevel = 0.01
minDistance = 5

def harris_corners(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    float_img = np.float32(img_grey)
    dest = cv2.cornerHarris(float_img, blockSize, kSize, k)
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

    corners = cv2.goodFeaturesToTrack(img_grey, maxCorners, qualityLevel, minDistance, useHarrisDetector=False)
    corners = np.intp(corners)

    return corners


def flann(img1, img2, des1, kp1, des2, kp2):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    goodMatches = []

    # ratio test as per Lowe's paper
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            goodMatches.append(m)

    minGoodMatches = 20
    if len(goodMatches) < minGoodMatches:
        srcPts = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

        errorThreshold = 5
        M, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, errorThreshold)
        matchesMask = mask.ravel().tolist()

        score = np.sum(mask) / len(mask)
        print(f'Score: {score}')

    else:
        print('No good matches')
        matchesMask = None

    drawParams = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    imgMatch = cv2.drawMatches(img1, kp1, img2, kp2, goodMatches, None, **drawParams)
    plt.figure()
    plt.imshow(imgMatch)
    plt.show()

    return



def sift_compare(img1, img2):
    grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(grey1, None)
    kp2, des2 = sift.detectAndCompute(grey2, None)

    flann(img1, img2, des1, kp1, des2, kp2)

    return


def harris_compare(img1, img2):
    grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    float1 = np.float32(grey1)
    float2 = np.float32(grey2)

    dest1 = cv2.cornerHarris(float1, blockSize, kSize, k)
    dest1 = cv2.dilate(dest1, None)
    ret1, dst_thresh1 = cv2.threshold(dest1, 0.01 * dest1.max(), 255, 0)
    dst_thresh1 = np.uint8(dst_thresh1)

    dest2 = cv2.cornerHarris(float2, blockSize, kSize, k)
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

    flann(img1, img2, des1, kp1, des2, kp2)
    return


def shi_tomasi_compare(img1, img2):
    grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    corners1 = cv2.goodFeaturesToTrack(grey1, maxCorners, qualityLevel, minDistance, useHarrisDetector=False)
    corners1 = np.intp(corners1)

    corners2 = cv2.goodFeaturesToTrack(grey2, maxCorners, qualityLevel, minDistance, useHarrisDetector=False)
    corners2 = np.intp(corners2)

    kps1 = [cv2.KeyPoint(x=float(x), y=float(y), size=3) for [[x, y]] in corners1]
    kps2 = [cv2.KeyPoint(x=float(x), y=float(y), size=3) for [[x, y]] in corners2]

    sift = cv2.SIFT_create()
    kp1, des1 = sift.compute(grey1, kps1)
    kp2, des2 = sift.compute(grey2, kps2)

    flann(img1, img2, des1, kp1, des2, kp2)
    return

