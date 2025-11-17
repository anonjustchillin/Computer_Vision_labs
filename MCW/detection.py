import cv2
import numpy as np
import matplotlib.pyplot as plt


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
        if m.distance < 0.76 * n.distance:
            goodMatches.append(m)

    goodMatches = [m for m in goodMatches if m.distance < 170]

    minGoodMatches = 10
    if len(goodMatches) >= minGoodMatches:
        srcPts = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

        errorThreshold = 5
        M, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, errorThreshold)
        matchesMask = mask.ravel().tolist()

        score = np.sum(mask) / len(mask)
        print(f'Score: {score}')
        print(f'Good matches: {len(goodMatches)}')

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


def shi_tomasi_compare(img1, img2, maxCorners, qualityLevel, minDistance):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    corners1 = cv2.goodFeaturesToTrack(img1, maxCorners, qualityLevel, minDistance, useHarrisDetector=False)
    corners1 = np.intp(corners1)
    #corners1 = cv2.cornerSubPix(img1, corners1, (5, 5), (-1, -1), criteria)
    #print("Corners1 detected:", len(corners1) if corners1 is not None else 0)

    corners2 = cv2.goodFeaturesToTrack(img2, maxCorners, qualityLevel, minDistance, useHarrisDetector=False)
    corners2 = np.intp(corners2)
    #corners2 = cv2.cornerSubPix(img2, corners2, (5, 5), (-1, -1), criteria)
    #print("Corners2 detected:", len(corners2) if corners2 is not None else 0)

    kps1 = [cv2.KeyPoint(x=float(x), y=float(y), size=3) for [[x, y]] in corners1]
    kps2 = [cv2.KeyPoint(x=float(x), y=float(y), size=3) for [[x, y]] in corners2]

    sift = cv2.SIFT_create()
    kp1, des1 = sift.compute(img1, kps1)
    kp2, des2 = sift.compute(img2, kps2)

    #print("SIFT keypoints1:", len(kp1))
    #print("Descriptors shape:", None if des1 is None else des1.shape)
    #print("SIFT keypoints2:", len(kp2))
    #print("Descriptors shape:", None if des2 is None else des2.shape)

    flann(img1, img2, des1, kp1, des2, kp2)
    return
