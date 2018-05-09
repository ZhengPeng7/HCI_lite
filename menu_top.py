import cv2
import numpy as np


def attach_menu(frame_bg, menu_dict, icon_len_side=80):
    """
    Description: Attach menu on the top of frame_bg.
                 To choose some function, just move your hand over the corresponding icon.
    Params:
        frame_bg: Input from web camera.
        menu_dict: Icons to be attached on the top of a frame_bg,
                   which consists of color panel, thickness regulator, and some icons with other speical usages.
        icon_len_side: The length of icon side.
    """
    for idx, icon in enumerate(menu_dict):
        val = menu_dict[icon]
        frame_bg[
            :icon_len_side, idx*icon_len_side:(idx+1)*icon_len_side
        ] = val if isinstance(val, tuple) else cv2.imread(val)

    return frame_bg
