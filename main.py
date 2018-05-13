import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import menu_top
import video_mode
import ball_tracking
import formula_calc
from config import args, args_menu, MODE, args_display, args_styleTransfer, args_calc


# scene settings
cv2.namedWindow('Amuse_park')
cv2.moveWindow('Amuse_park', 300, 20)

# preparation for tool variables
cap = cv2.VideoCapture(args["video_source"])
hei_frame, wid_frame = cap.read()[1].shape[:2]
frame_fg = np.zeros((hei_frame, wid_frame, 3), dtype=np.uint8)  # front frame

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'FLV1')
video_writer = cv2.VideoWriter(
    'formula_evaluation.flv',
    fourcc,
    20.0,
    (800, 600)
)
# Video flow
while cap.isOpened():
    iteration_start = time.time()
    ret, frame_bg = cap.read()
    cv2.flip(frame_bg, 1, frame_bg)

    # Turn into styleTransfer mode needs several secs to let move away from cap
    if MODE == "styleTransfer" and args_styleTransfer["waiting_sec"] > 0:
        args_styleTransfer["waiting_sec"] -= 1
        frame_waiting_before_styleTransfer = np.random.randint(
            0, 256, (600, 1300, 3), dtype=np.uint8
        )
        cv2.putText(
            frame_waiting_before_styleTransfer,
            "Please move away from the screen.",
            (32, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4
        )
        cv2.putText(
            frame_waiting_before_styleTransfer,
            ''.join(
                (
                    "Style transfering will start in ",
                    str(args_styleTransfer["waiting_sec"]),
                    ('seconds' if args_styleTransfer["waiting_sec"] > 1 else 'second'),
                    '...'
                )
            ),
            (32, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4
        )
        cv2.imshow(
            'Amuse_park',
            frame_waiting_before_styleTransfer
        )
        cv2.waitKey(1000)
        args_styleTransfer["waiting_sec"] -= 1
        continue


    # The most essential part, ranging different modes
    frame_bg, frame_fg, MODE = video_mode.video_mode(frame_bg, frame_fg, MODE, eval('args_'+MODE))

    mask_fg = frame_fg > 0

    if MODE == 'display':
        # Attachment of menu on the top of frame_bg
        frame_bg_with_menu = menu_top.attach_menu(
            frame_bg.copy(), args_menu["menu_dict"], args_menu["icon_len_side"]
        )
        frame = np.add(
            np.multiply(frame_fg, mask_fg),
            np.multiply(frame_bg_with_menu, ~mask_fg)
        )
    else:
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
    video_writer.write(frame)
    cv2.imshow('Amuse_park', frame)

    # key pressed
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    # elif key == ord("g"):
    #     frame_fg = np.zeros((hei_frame, wid_frame, 3), dtype=np.uint8)
    #     MODE = "gaming"
    elif key == ord("d"):
        frame_fg = np.zeros((hei_frame, wid_frame, 3), dtype=np.uint8)
        MODE = "display"
    elif key == ord("c"):
        frame_fg = np.zeros((hei_frame, wid_frame, 3), dtype=np.uint8)
        args_calc["res"] = ''
        MODE = "calc"
    elif key == ord("t"):
        frame_fg = np.zeros((hei_frame, wid_frame, 3), dtype=np.uint8)
        MODE = "styleTransfer"
    elif key == ord("e"):
        args_calc["res"] = formula_calc.evaluation(frame_fg, args_calc)
    else:
        pass

# THE END
cap.release()
video_writer.release()
cv2.destroyAllWindows()
