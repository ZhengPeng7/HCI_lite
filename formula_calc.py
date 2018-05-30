import cv2
import numpy as np
from PIL import Image
import pytesseract


def formula_calc(frame_bg, frame_fg, args_calc):
    if args_calc['res'] != '':
        cv2.putText(
            frame_fg,
            'The result equals: ',
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (64, 1, 32),
            7 
        )
        cv2.putText(
            frame_fg,
            str(args_calc['res']),
            (250, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (64, 1, 32),
            9 
        )
        return frame_bg, frame_fg
    
    blurred = cv2.GaussianBlur(frame_bg, (11, 11), 0)
    hsv = cv2.cvtColor(frame_bg, cv2.COLOR_BGR2HSV)
    mask_color = cv2.inRange(hsv, args_calc["orange_lower"], args_calc["orange_upper"])
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
            cv2.circle(frame_bg, (int(x), int(y)), int(radius),
                tuple([int(co) for co in np.random.randint(0, 256, (3,))]), 2)
            cv2.circle(frame_fg, center, 5, args_calc["font_color"], -1)
            args_calc["pts"].appendleft(center)
    else:
        for i in range(len(args_calc['pts'])):
            args_calc['pts'].appendleft((0, 0))
    for i in np.arange(1, len(args_calc["pts"])):
        # Achieve fluent writing!
        thickness = 7
        if np.linalg.norm(
            np.asarray(args_calc["pts"][i-1]) - np.asarray(args_calc["pts"][i])
        ) < 100:
            cv2.line(
                frame_fg,
                args_calc["pts"][i - 1], args_calc["pts"][i],
                args_calc["font_color"],
                thickness
            )


    cv2.putText(
        frame_bg,
        'Press "e" to evaluate the formula.',
        (20, 150),
        cv2.FONT_HERSHEY_COMPLEX,
        1, (0, 0, 255), 3
    )
    frame_fg[:33, :33] = 0
    return frame_bg, frame_fg


def evaluation(frame_fg, args_calc):
    formula = cv2.threshold(
        cv2.cvtColor(frame_fg, cv2.COLOR_BGR2GRAY),
        1, 255, cv2.THRESH_BINARY_INV
    )[1]
    formula = cv2.morphologyEx(
        formula,
        cv2.MORPH_OPEN,
        args_calc["kernel_open"]
    )

    cv2.imwrite('./images/formula.jpg', 255 - formula)
    formula = Image.open('./images/formula.jpg')
    res = pytesseract.pytesseract.image_to_string(formula)
    print(res)
    if res != '':
        res = eval(res)
    return res

