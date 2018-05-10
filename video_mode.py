import cv2
import numpy as np
import ball_tracking


def video_mode(frame_fg, frame_bg, mode, args):
    """
    Description: Do some operations on frame.
    Params:
        frame_fg: Paintings layer.
        frame_bg: Input from web camera.
        mode: Choose one way to change the frame.
    """
    frame_fg = eval('mode_' + mode)(frame_fg, frame_bg, args)

    return frame_fg


def mode_display(frame_fg, frame_bg, args_display):
    """
    Description: In display mode. Main function is the effect of tailing.
    Params:
        frame_fg: The canvas layer.
        frame_bg: The background layer.
        args_display: The arguments used in display mode.
    """
    (frame_fg, args_display) = ball_tracking.ball_tracking(
        frame_bg, np.zeros_like(frame_fg), args_display
    )
    return frame_fg


def mode_writing(frame_fg, frame_bg, args_writing):

    return frame_fg


def mode_gaming(frame_fg, frame_bg, args_gaming):
    return frame_fg


def mode_calc(frame_fg, frame_bg, args_calc):
    return frame_fg
