import cv2
import numpy as np
import ball_tracking
import style_transfer
import cut_grab
import wear_glasses

def video_mode(frame_bg, frame_fg, mode, args):
    """
    Description: Do some operations on frame.
    Params:
        frame_bg: Input from web camera.
        frame_fg: Paintings layer.
        mode: Choose one way to change the frame.
    """
    if mode == 'display':
        frame_bg, frame_fg, mode = eval('mode_' + mode)(frame_bg, frame_fg, args, mode)
    else:
        frame_bg, frame_fg = eval('mode_' + mode)(frame_bg, frame_fg, args)

    return frame_bg, frame_fg, mode


def mode_display(frame_bg, frame_fg, args_display, mode):
    """
    Description: In display mode. Main function is to show the effect of tailing.
    Params:
        frame_bg: The background layer.
        frame_fg: The canvas layer.
        args_display: The arguments used in display mode.
        mode: Current mode, coz display mode can turn into other modes.
    """
    frame_fg, args_display, mode = ball_tracking.ball_tracking(
        frame_bg, np.zeros_like(frame_fg), args_display, mode
    )
    return frame_bg, frame_fg, mode


def mode_grabCut(frame_bg, frame_fg, args_grabCut):
    """
    Description: In grabCut mode. Main function is to cut the grab...
    Params:
        frame_bg: The background layer.
        frame_fg: The canvas layer.
        args_grabCut: The arguments used in args_grabCut mode.
    """
    frame_bg = cut_grab.cut_grab(frame_bg, args_grabCut)

    return frame_bg, frame_fg


def mode_styleTransfer(frame_bg, frame_fg, args_styleTransfer):
    """
    Description: In styleTransfer mode, it transfers the style of input into
                 the style of Starry Night by Vincent Willem van Gogh.
    Params:
        frame_bg: The background layer.
        frame_fg: The canvas layer.
        args_styleTransfer: The arguments used in styleTransfer mode.
    """
    frame_bg = style_transfer.style_transfer(
        frame_bg, args_styleTransfer
    )
    
    return frame_bg, frame_fg


def mode_glass(frame_bg, frame_fg, args_glass):
    """
    Description: In glass mode, your eyes will be located with a pair of glass.
    Params:
        frame_bg: The background layer.
        frame_fg: The canvas layer.
        args_glass: The arguments used in glass mode.
    """
    frame_bg = wear_glasses.wear_glasses(
        frame_bg, args_glass
    )
    return frame_bg, frame_fg


def mode_AR(frame_bg, frame_fg, args_AR):
    """
    Description: Modified from opencv sample -- plane_ar.
                 In this mode, you can select a rect to build a roof on it.
    Params:
        frame_bg: The background layer.
        frame_fg: The canvas layer.
        args_AR: The arguments used in AR mode.
    """
    frame_bg = args_AR['app_ar'].run(
        frame_bg
    )
    return frame_bg, frame_fg
