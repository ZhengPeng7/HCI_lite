import numpy as np
import cv2

cap = cv2.VideoCapture('vtest.avi')

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)

    cv2.imshow('res', cv2.resize(fgmask, (666, 444), cv2.INTER_CUBIC))
    cv2.imshow('src', cv2.resize(frame, (666, 444), cv2.INTER_CUBIC))
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

