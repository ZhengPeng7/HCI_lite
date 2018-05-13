import cv2
import numpy as np
import ball_tracking
import style_transfer


def video_mode(frame_bg, frame_fg, mode, args):
    """
    Description: Do some operations on frame.
    Params:
        frame_fg: Paintings layer.
        frame_bg: Input from web camera.
        mode: Choose one way to change the frame.
    """
    if mode == 'display':
        frame_bg, frame_fg, mode = eval('mode_' + mode)(frame_bg, frame_fg, args, mode)
    else:
        frame_bg, frame_fg = eval('mode_' + mode)(frame_bg, frame_fg, args)

    return frame_bg, frame_fg, mode


def mode_display(frame_bg, frame_fg, args_display, mode):
    """
    Description: In display mode. Main function is the effect of tailing.
    Params:
        frame_fg: The canvas layer.
        frame_bg: The background layer.
        args_display: The arguments used in display mode.
    """
    (frame_fg, args_display) = ball_tracking.ball_tracking(
        frame_bg, np.zeros_like(frame_fg), args_display, mode
    )
    return frame_bg, frame_fg, mode


# def mode_writing(frame_bg, frame_fg, args_writing):

#     return frame_bg, frame_fg


# def mode_gaming(frame_bg, frame_fg, args_gaming):
#     return frame_bg, frame_fg


def mode_calc(frame_bg, frame_fg, args_calc):
    return frame_bg, frame_fg


def mode_styleTransfer(frame_bg, frame_fg, args_styleTransfer):
    frame_bg = style_transfer.style_transfer(
        frame_bg, args_styleTransfer
    )
    
    return frame_bg, frame_fg
