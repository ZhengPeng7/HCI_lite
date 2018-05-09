import cv2
import numpy as np


# argument settings
args = {
    "video_source": 0,
}

# tool variable preparation
cap = cv2.VideoCapture(args["video_source"])
hei_frame, wid_frame = cap.read()[1].shape[:2]

# scene settings
cv2.namedWindow('Amuse_park')
cv2.moveWindow('Amuse_park', 100, 20)

# Video flow
while cap.isOpened():
    ret, frame_bg = cap.read()

    # default setting
    frame_fg = np.zeros_like(frame_bg)
    mask_fg = frame_fg > 0

    frame = cv2.add(np.multiply(frame_fg, mask_fg), np.multiply(frame_bg, ~mask_fg))
    cv2.imshow('Amuse_park', cv2.resize(cv2.flip(frame, 1), (800, 600)))
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

# THE END
cap.release()
cv2.destroyAllWindows()
