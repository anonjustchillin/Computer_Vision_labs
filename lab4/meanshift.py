import cv2
from random import randint
import numpy as np


def meanShift(video):
    success, frame = video.read()

    rois = []
    roi_hists = []
    track_wins = []
    color_list = []

    term_crit = (cv2.TERM_CRITERIA_EPS |
                 cv2.TERM_CRITERIA_COUNT, 15, 2)

    if success:
        while True:
            print("Select a desired area.")
            print('Press any other key to select the next object')
            print('or press "q" to start object tracking.')

            bbox = cv2.selectROI('Select ROI', frame, fromCenter=False)
            if bbox == (0, 0, 0, 0):
                continue
            x, y, w, h = [int(box) for box in bbox]
            roi = frame[y:y + h, x:x + w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi,
                               np.array((0., 60., 32.)),
                               np.array((180., 255., 255.)))
            roi_hist = cv2.calcHist([hsv_roi],[0], mask, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

            rois.append(roi)
            roi_hists.append(roi_hist)
            track_wins.append((x, y, w, h))

            color_list.append((randint(0, 255),
                               randint(0, 255),
                               randint(0, 255)))

            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                break
        print('Selected bounding boxes {}'.format(track_wins))
        cv2.destroyAllWindows()
        print('Tracking...')

        while video.isOpened():
            success, frame = video.read()

            if success:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                for i, roi_hist in enumerate(roi_hists):
                    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

                    ret, track_wins[i] = cv2.meanShift(dst, track_wins[i], term_crit)

                    pts = cv2.boxPoints(ret)
                    pts = np.intp(pts)
                    cv2.polylines(frame, [pts], True, color_list[i], 3)

                cv2.imshow('MeanShift', frame)

                k = cv2.waitKey(1) & 0xFF
                if k == 27:  # escape
                    break
            else:
                break

    return