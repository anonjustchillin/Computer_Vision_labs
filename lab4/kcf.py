import cv2
from random import randint


def createTracker():
    return cv2.legacy.TrackerKCF_create()


def kcf(video):
    multi_tracker = cv2.legacy.MultiTracker_create()

    success, frame = video.read()

    bounding_box_list = []
    color_list = []

    if success:
        while True:
            print("Select a desired area.")
            print('Press any other key to select the next object')
            print('or press "q" to start object tracking.')

            bounding_box = cv2.selectROI('MultiTracker', frame, False, False)
            bounding_box_list.append(bounding_box)

            color_list.append((randint(0, 255),
                               randint(0, 255),
                               randint(0, 255)))

            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                break
        print('Selected bounding boxes {}'.format(bounding_box_list))
        cv2.destroyAllWindows()
        print('Tracking...')

        for bbox in bounding_box_list:
            if bbox is None:
                continue
            multi_tracker.add(createTracker(), frame, bbox)

        while video.isOpened():
            success, frame = video.read()
            if success:
                success, boxes = multi_tracker.update(frame)
                for i, bbox in enumerate(boxes):
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, color_list[i], 3)
                cv2.imshow('KCF', frame)

                k = cv2.waitKey(1) & 0xFF
                if k == 27:  # escape
                    break
            else:
                break

    return