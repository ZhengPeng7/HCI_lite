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
    # mask_clothes = clothes_extraction(frame_bg)   # shape = (480, 640)
    # print(frame_bg_stylized.shape, mask_clothes.shape)
    # frame_bg = np.add(
    #     np.multiply(frame_bg_stylized, mask_clothes),
    #     np.multiply(
    #         cv2.resize(
    #             frame_bg,
    #             args_styleTransfer["content_target_resize"[2:0:-1]],
    #             cv2.INTER_CUBIC
    #         ),
    #         ~mask_clothes
    #     )
    # )

    return frame_bg_stylized


def clothes_extraction(img):
    # person = cv2.grabCut(
    #     img,
    #     mask=np.ones_like(img),
    #     rect=,
    #     bgdModel=,
    #     fgdModel=,
    #     iterCount=
    # )
    return np.ones((img.shape[:2]), dtype=np.bool)


def skin_extraction(img):
    
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    img = ((cv2.GaussianBlur(skinMask, (3, 3), 0) > 0) * 255).astype(np.uint8)
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[:3]
    img = np.zeros_like(img)
    for i in range(len(cnts)):
        cv2.drawContours(img, cnts, i, 255, cv2.FILLED)
    
    return skin_mask
