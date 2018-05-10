import cv2
import numpy as np
import menu_top
import video_mode
import ball_tracking
from config import args, args_menu, MODE, args_display


# scene settings
cv2.namedWindow('Amuse_park')
cv2.moveWindow('Amuse_park', 100, 20)

# preparation for tool variables
cap = cv2.VideoCapture(args["video_source"])
hei_frame, wid_frame = cap.read()[1].shape[:2]
frame_fg = np.zeros((hei_frame, wid_frame, 3), dtype=np.uint8)  # front frame



# Video flow
while cap.isOpened():
    ret, frame_bg = cap.read()
    cv2.flip(frame_bg, 1, frame_bg)

    # Attachment of menu on the top of frame_bg
    frame_bg_with_menu = menu_top.attach_menu(
        frame_bg.copy(), args_menu["menu_dict"], args_menu["icon_len_side"]
    )

    # The most essential part, ranging different modes
    frame_fg = video_mode.video_mode(frame_fg, frame_bg, MODE, eval('args_'+MODE))
    mask_fg = frame_fg > 0


    frame = np.add(
        np.multiply(frame_fg, mask_fg),
        np.multiply(frame_bg_with_menu, ~mask_fg)
    )
    cv2.imshow('Amuse_park', cv2.resize(frame, (800, 600)))

    # key pressed
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('w'):
        MODE = "writing"
    elif key == ord("g"):
        MODE = "gaming"
    elif key == ord("d"):
        MODE = "display"
    elif key == ord("c"):
        MODE = "calc"
    else:
        pass

# THE END
cap.release()
cv2.destroyAllWindows()
