import cv2
import numpy as np
import menu_top


# argument settings
args = {
    "video_source": 0,
}

# menu_top settings
menu_dict = {
    "color_purpleaa": (255, 255, 255),
    "color_redaa": (0, 0, 0),
    "minus_icon": "images/minus_icon.png",
    "plus_icon": "images/plus_icon.png",
    "color_purple": (161, 0, 161),
    "color_red": (1, 1, 253),
    "color_green": (1, 253, 1),
    "color_blue": (253, 1, 1),
}

# scene settings
cv2.namedWindow('Amuse_park')
cv2.moveWindow('Amuse_park', 100, 20)

# tool variable preparation
cap = cv2.VideoCapture(args["video_source"])
hei_frame, wid_frame = cap.read()[1].shape[:2]

# Video flow
while cap.isOpened():
    ret, frame_bg = cap.read()

    frame_bg = menu_top.attach_menu(frame_bg, menu_dict)

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
