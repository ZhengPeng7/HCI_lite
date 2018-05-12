import cv2
import numpy as np
import time
import menu_top
import video_mode
import ball_tracking
from config import args, args_menu, MODE, args_display, args_styleTransfer


# scene settings
cv2.namedWindow('Amuse_park')
cv2.moveWindow('Amuse_park', 300, 20)

# preparation for tool variables
cap = cv2.VideoCapture(args["video_source"])
hei_frame, wid_frame = cap.read()[1].shape[:2]
frame_fg = np.zeros((hei_frame, wid_frame, 3), dtype=np.uint8)  # front frame



# Video flow
while cap.isOpened():
    iteration_start = time.time()
    ret, frame_bg = cap.read()
    cv2.flip(frame_bg, 1, frame_bg)


    # The most essential part, ranging different modes
    frame_bg, frame_fg = video_mode.video_mode(frame_bg, frame_fg, MODE, eval('args_'+MODE))

    mask_fg = frame_fg > 0

    if MODE == 'display':
        print(frame_bg.shape, mask_fg.shape)
        # Attachment of menu on the top of frame_bg
        frame_bg_with_menu = menu_top.attach_menu(
            frame_bg.copy(), args_menu["menu_dict"], args_menu["icon_len_side"]
        )
        frame = np.add(
            np.multiply(frame_fg, mask_fg),
            np.multiply(frame_bg_with_menu, ~mask_fg)
        )
    else:
        print(frame_bg.shape, mask_fg.shape)
        frame = np.add(
            np.multiply(frame_fg, mask_fg),
            np.multiply(
                cv2.resize(
                    frame_bg,
                    mask_fg.shape[1::-1],
                    interpolation=cv2.INTER_CUBIC
                ),
                ~mask_fg
            )
        )
    fps = str(round(1 / (time.time() - iteration_start), 2))
    frame = cv2.resize(frame, (800, 600))
    cv2.putText(frame, 'FPS: '+fps, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 255), 3)
    cv2.imshow('Amuse_park', frame)

    # key pressed
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('w'):
        frame_fg = np.zeros((hei_frame, wid_frame, 3), dtype=np.uint8)
        MODE = "writing"
    elif key == ord("g"):
        frame_fg = np.zeros((hei_frame, wid_frame, 3), dtype=np.uint8)
        MODE = "gaming"
    elif key == ord("d"):
        frame_fg = np.zeros((hei_frame, wid_frame, 3), dtype=np.uint8)
        MODE = "display"
    elif key == ord("c"):
        frame_fg = np.zeros((hei_frame, wid_frame, 3), dtype=np.uint8)
        MODE = "calc"
    elif key == ord("t"):
        frame_fg = np.zeros((hei_frame, wid_frame, 3), dtype=np.uint8)
        MODE = "styleTransfer"
    else:
        pass

# THE END
cap.release()
cv2.destroyAllWindows()
