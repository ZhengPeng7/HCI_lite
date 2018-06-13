import numpy as np
import cv2 as cv
import cv2
import matplotlib.pyplot as plt


def extract_pikachu():

    img_ori = img.copy()
    img_masked = img.copy()

    mask = np.zeros(img_masked.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # rect = (300, 10, 950, 700)
    rect = (X_ul, Y_ul, X_dr, Y_dr)
    cv2.rectangle(img_ori, rect[:2], rect[2:], color=(161, 0, 128), thickness=3)
    cv2.grabCut(img_masked, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    img_masked = img_masked*mask2[:, :, np.newaxis]

    plt.figure(figsize=(18, 8))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB))
    plt.show()


drawing = False
draw_box = False
x_ul, y_ul = -1, -1
x_dr, y_dr = -1, -1
# mouse callback function
def draw_rect(event, x, y, flags, param):
    global x_ul, y_ul, x_dr, y_dr, drawing, img_draw, img, X_dr, Y_dr, X_ul, Y_ul, draw_box
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_ul, y_ul = x, y
        x_dr, y_dr = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            draw_box = True
            x_dr, y_dr = x, y
            X_dr, Y_dr = max(x, x_ul), max(y, y_ul)
            X_ul, Y_ul = min(x, x_ul), min(y, y_ul)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_dr, y_dr = x, y
        X_dr, Y_dr = max(x, x_ul), max(y, y_ul)
        X_ul, Y_ul = min(x, x_ul), min(y, y_ul)
        if (Y_dr - Y_ul) > 20 and (X_dr - X_ul) > 20:
            extract_pikachu()

img = cv2.imread('./pikachu.jpg')
img_draw = img.copy()
cv2.namedWindow('pikachu')
cv2.setMouseCallback('pikachu', draw_rect)
while True:
    if draw_box:
        img_draw[Y_ul:Y_dr, X_ul-2:X_ul+2] = (255, 0, 0)
        img_draw[Y_ul:Y_dr, X_dr-2:X_dr+2] = (0, 255, 0)
        img_draw[Y_ul-2:Y_ul+2, X_ul:X_dr] = (0, 0, 255)
        img_draw[Y_dr-2:Y_dr+2, X_ul:X_dr] = (161, 0, 128)
    cv2.imshow('pikachu', img_draw)
    img_draw = img.copy()
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
cv2.destroyAllWindows()
