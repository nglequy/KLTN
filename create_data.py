import os.path

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.SelfiSegmentationModule import SelfiSegmentation

cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)

success_bG, background = cam.read()
bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
segmentor = SelfiSegmentation()
detector = HandDetector(detectionCon=0.8, maxHands=2)

i = 0
label = "max"

def remove_background(frame):
    mask = bgModel.apply(frame, learningRate=0)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    return res


def hand_process(img_hand, i):
    img_hand = cv2.resize(img_hand, dsize=(400, 400))
    img_hand = cv2.cvtColor(img_hand, cv2.COLOR_BGR2GRAY)
    img_hand = cv2.GaussianBlur(img_hand, (7, 7), 0)
    thresh, img_hand = cv2.threshold(img_hand, 25, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Hands", img_hand)
    if 60 <= i <= 1060:
        print("So anh capture: ", i-60)
        if not os.path.exists('data/' + str(label)):
            os.mkdir(('data/' + str(label)))
        cv2.imwrite('data/' + str(label) + "/" + str(i) + ".png", img_hand)


while cam.isOpened():

    success, img = cam.read()
    img = cv2.bilateralFilter(img, 5, 50, 100)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    hands = detector.findHands(img, draw=False)
    img_hand = remove_background(img)
    img_hand = cv2.blur(img_hand, (9, 9))
    if hands:
        i = i+1
        print(i)
        x, y, w, h = hands[0]['bbox']
        if w > h:
            cv2.rectangle(img, (x - 10, int(y + h / 2 - w / 2 - 10)), (x + w + 10, int(y + h / 2 + w / 2 + 10)),
                          (0, 255, 255), 2)
            if (10 < x) & (0 < y + h / 2 - w / 2 - 10):
                img_hand = img_hand[int(y + h / 2 - w / 2 - 10):int(y + h / 2 + w / 2 + 10), x - 10:x + w + 10]
                hand_process(img_hand, i)
        else:
            cv2.rectangle(img, (int(x + w / 2 - h / 2 - 10), y - 10), (int(x + w / 2 + h / 2 + 10), y + h + 10),
                          (0, 255, 255), 2)
            if (0 < x + w / 2 - h / 2 - 10) & (10 < y):
                img_hand = img_hand[y - 10:y + h + 10, int(x + w / 2 - h / 2 - 10):int(x + w / 2 + h / 2 + 10)]
                hand_process(img_hand, i)

    cv2.imshow("Image", img)

    cv2.waitKey(1)
