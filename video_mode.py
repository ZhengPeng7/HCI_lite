import cv2
import numpy as np
import ball_tracking
import style_transfer
import formula_calc
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


# def mode_gaming(frame_bg, frame_fg, args_gaming):
#     return frame_bg, frame_fg


def mode_calc(frame_bg, frame_fg, args_calc):
    """
    Description: In calc mode. Main function is to calculate the hand written formula.
    Params:
        frame_bg: The background layer.
        frame_fg: The canvas layer.
        args_calc: The arguments used in args_calc mode.
    """
    frame_bg, frame_fg = formula_calc.formula_calc(frame_bg, frame_fg, args_calc)
    # cv2.imshow('fg', frame_fg)
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
    frame_bg = wear_glasses.wear_glasses(
        frame_bg, args_glass
    )
    return frame_bg, frame_fg
