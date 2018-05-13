import cv2
import numpy as np


def style_transfer(frame_bg, args_styleTransfer):
    img = cv2.resize(
        frame_bg,
        args_styleTransfer["content_target_resize"][2:0:-1],
        interpolation=cv2.INTER_CUBIC,
    )

    img_4d = img[np.newaxis, :]

    frame_bg_stylized = args_styleTransfer["sess"].run(
        args_styleTransfer["Y"],
        feed_dict={args_styleTransfer["X"]: img_4d}
    )
    frame_bg_stylized_BGR = cv2.resize(
        np.squeeze(frame_bg_stylized).astype(np.uint8),
        args_styleTransfer["content_target_resize"][2:0:-1],
        cv2.INTER_CUBIC
    )
    frame_bg_stylized_RGB = cv2.cvtColor(frame_bg_stylized_BGR, cv2.COLOR_BGR2RGB)
    frame_bg_stylized = np.add(
        np.multiply(frame_bg_stylized_BGR, ~args_styleTransfer["mask_rgb"]),
        np.multiply(frame_bg_stylized_RGB, args_styleTransfer["mask_rgb"])
    )
    mask_clothes = clothes_extraction(frame_bg, args_styleTransfer) > 0
    mask_clothes = cv2.cvtColor(
        cv2.resize(mask_clothes.astype(np.uint8)*255, args_styleTransfer["content_target_resize"][2:0:-1]),
        cv2.COLOR_GRAY2BGR
    ) > 0
    if args_styleTransfer["show_detail"]:
        cv2.imshow('mask_clothes', mask_clothes.astype(np.uint8)*255)
    frame_bg = np.add(
        np.multiply(frame_bg_stylized, mask_clothes),
        np.multiply(
            cv2.resize(
                frame_bg,
                args_styleTransfer["content_target_resize"][2:0:-1],
                cv2.INTER_CUBIC
            ),
            ~mask_clothes
        )
    )


    return frame_bg


def clothes_extraction(frame_bg, args_styleTransfer):
    mask_figure = figure_extraction(frame_bg, args_styleTransfer)
    mask_figure = mask_figure.astype(np.uint8)
    cnts = cv2.findContours(mask_figure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    for i in range(len(cnts)-1, -1, -1):
        if cv2.contourArea(cnts[i]) < 289:
            cnts.pop(i)
    mask_figure = np.zeros_like(mask_figure)
    for i in range(len(cnts)):
        cv2.drawContours(mask_figure, cnts, i, 255, cv2.FILLED)
    mask_skin = skin_extraction(frame_bg, args_styleTransfer)
    mask_figure_skin_interaction = cv2.bitwise_and(mask_figure, mask_skin)
    mask_clothes = np.subtract(
        mask_figure,
        mask_figure_skin_interaction
    ) > 0
    if args_styleTransfer["show_detail"]:
        cv2.imshow('mask_figure', mask_figure)
        cv2.imshow('mask_skin', mask_skin)
    return mask_clothes


def figure_extraction(frame_bg, args_styleTransfer):
    mask_figure = args_styleTransfer["fgbg"].apply(frame_bg)
    mask_figure = cv2.morphologyEx(
        mask_figure,
        cv2.MORPH_OPEN,
        args_styleTransfer["kernel_open"]
    )

    return mask_figure


def skin_extraction(frame_bg, args_styleTransfer):
    
    converted = cv2.cvtColor(frame_bg, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, args_styleTransfer["skin_lower"], args_styleTransfer["skin_upper"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    img = ((cv2.GaussianBlur(skinMask, (3, 3), 0) > 0) * 255).astype(np.uint8)
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[:3]
    mask_skin = np.zeros_like(img)
    for i in range(len(cnts)):
        cv2.drawContours(mask_skin, cnts, i, 255, cv2.FILLED)
    
    return mask_skin
