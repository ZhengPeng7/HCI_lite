import cv2
import numpy as np
import ball_tracking


def video_mode(frame_bg, frame_fg, mode, args):
    """
    Description: Do some operations on frame.
    Params:
        frame_fg: Paintings layer.
        frame_bg: Input from web camera.
        mode: Choose one way to change the frame.
    """
    frame_bg, frame_fg = eval('mode_' + mode)(frame_bg, frame_fg, args)

    return frame_bg, frame_fg


def mode_display(frame_bg, frame_fg, args_display):
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
    return frame_bg, frame_fg


def mode_writing(frame_bg, frame_fg, args_writing):

    return frame_bg, frame_fg


def mode_gaming(frame_bg, frame_fg, args_gaming):
    return frame_bg, frame_fg


def mode_calc(frame_bg, frame_fg, args_calc):
    return frame_bg, frame_fg


def mode_styleTransfer(frame_bg, frame_fg, args_styleTransfer):
    
    img_4d = cv2.resize(
        frame_bg,
        args_styleTransfer["content_target_resize"][2:0:-1],
        interpolation=cv2.INTER_CUBIC,
    )[np.newaxis, :]

    frame_bg = args_styleTransfer["sess"].run(
        args_styleTransfer["Y"],
        feed_dict={args_styleTransfer["X"]: img_4d}
    )
    frame_bg = np.squeeze(frame_bg).astype(np.uint8)
    frame_bg = cv2.cvtColor(frame_bg, cv2.COLOR_BGR2RGB)
    return frame_bg, frame_fg
