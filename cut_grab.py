import cv2
import numpy as np


def cut_grab(frame_bg, args_grabCut):
    """
    Description: Extract your figure from the whole scene and put it into a new video.
    Params:
        frame_bg: Frame of background and you except for the top menu part.
        args_grabCut: Arguments needed in the grabCut mode.
    """
    frame_bg = cv2.cvtColor(frame_bg, cv2.COLOR_BGR2RGB)
    frame_bg = cv2.resize(
        frame_bg, (frame_bg.shape[1]//args_grabCut["ratio"],
        frame_bg.shape[0]//args_grabCut["ratio"]), cv2.INTER_CUBIC
    )
    mask = np.zeros(frame_bg.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = tuple(np.array([100, 50, 400, 400]) // args_grabCut["ratio"])

    cv2.grabCut(frame_bg, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), (0,), (1,)).astype('uint8')
    frame_bg = frame_bg * mask2[:, :, np.newaxis]

    # extract the background
    background = args_grabCut["bg_capture"].read()[1]

    # combine the figure and background using mask instead of iteration
    frame_bg = cv2.resize(
        cv2.cvtColor(frame_bg, cv2.COLOR_RGB2BGR), (480, 360), cv2.INTER_CUBIC
    )
    # return frame_bg, background
    mask_1 = frame_bg > 0
    frame_bg = frame_bg * mask_1 + background * ~mask_1

    return frame_bg
