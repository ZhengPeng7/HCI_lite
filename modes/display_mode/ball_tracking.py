import cv2
import numpy as np
from config import args, args_menu


def ball_tracking(frame_bg_without_menu, frame_fg, args_display, MODE):
    """
    Description: To describe the scene in the tracking mode. Since in the
                 tracking mode,there is no other operations besides tracking
                 (Clearing is not taken into account in this situation). So,
                 drawing the tracks in the frame_bg is just Okay and simplest.
    Params:
        frame_bg_without_menu: Frame of background except for the top menu part.
        frame_fg: Frame of canvas.
        args_display: Arguments needed in the tracking mode.
    """
    (dX, dY) = args_display["dXY"]
    blurred = cv2.GaussianBlur(frame_bg_without_menu, (11, 11), 0)
    hsv = cv2.cvtColor(frame_bg_without_menu, cv2.COLOR_BGR2HSV)
    mask_color = cv2.inRange(hsv, args["orange_lower"], args["orange_upper"])
    mask_color = cv2.erode(mask_color, None, iterations=2)
    mask_color = cv2.dilate(mask_color, None, iterations=2)
    cnts = cv2.findContours(
        mask_color.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            if args_display["is_start"] == 1:
                args_display["drawing_color"] = tuple(
                    [int(co) for co in np.random.randint(0, 256, (3,))]
                )
            # move into top menu
            if center[1] < args_menu["icon_len_side"]:
                if center[0] < args_menu["icon_len_side"] * 3:
                    args_display["drawing_color"] = list(
                        args_menu["menu_dict"].values()
                    )[center[0] // args_menu["icon_len_side"]]
                    args_display["is_start"] = 0
                elif center[0] < args_menu["icon_len_side"] * 4:
                    args_display["thick_coeff"] = min(17.0, args_display["thick_coeff"] * 1.03)
                elif center[0] < args_menu["icon_len_side"] * 5:
                    args_display["thick_coeff"] = max(1.7, args_display["thick_coeff"] / 1.03)
                elif center[0] < args_menu["icon_len_side"] * 6:
                    MODE = "grabCut"
                    return frame_fg, args_display, MODE
                elif center[0] < args_menu["icon_len_side"] * 7:
                    MODE = 'styleTransfer'
                    return frame_fg, args_display, MODE
                elif center[0] < args_menu["icon_len_side"] * 8:
                    MODE = 'glass'
                    return frame_fg, args_display, MODE
                else:
                    pass
            cv2.circle(frame_fg, (int(x), int(y)), int(radius),
                tuple([int(co) for co in np.random.randint(0, 256, (3,))]), 2)     # args_menu["menu_dict"]["color_red"]
            cv2.circle(frame_fg, center, 5, args_display["drawing_color"], -1)
            args_display["pts"].appendleft(center)
    for i in np.arange(1, len(args_display["pts"])):
        if args_display["pts"][i - 1] is None or args_display["pts"][i] is None:
            continue
        if args_display["counter"] >= 10 and i == 1 and args_display["pts"][-10] is not None:
            dX = args_display["pts"][-10][0] - args_display["pts"][i][0]
            dY = args_display["pts"][-10][1] - args_display["pts"][i][1]
            (dirX, dirY) = ("", "")
            if np.abs(dX) > 20:
                dirX = "East" if np.sign(dX) == 1 else "West"
            if np.abs(dY) > 20:
                dirY = "North" if np.sign(dY) == 1 else "South"
            if dirX != "" and dirY != "":
                args_display["direction"] = "{}-{}".format(dirY, dirX)
            else:
                args_display["direction"] = dirX if dirX != "" else dirY
        thickness = int(np.sqrt(args["deque_buffer"] / float(i + 1)) * args_display["thick_coeff"])
        cv2.line(frame_fg, args_display["pts"][i - 1], args_display["pts"][i], args_display["drawing_color"], thickness)
    cv2.putText(frame_fg, args_display["direction"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, args_display["drawing_color"], 3)
    cv2.putText(frame_fg, "dx: {}, dy: {}".format(dX, dY),
        (10, frame_fg.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, args_display["drawing_color"], 1)

    return frame_fg, args_display, MODE
