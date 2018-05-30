import cv2
import numpy as np


def extract_chars(image):
    # initial seed would be at (0, 0) and make it white
    if isinstance(image, str):
        gray = cv2.imread(image, 0)
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
    gray[:20, :20] = 255
    cv2.imshow('', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # get rid of margin
    for r in range(gray.shape[0]):
        if gray[r, :].any():
            top = max(r - 10, 0)
            break
    for r in range(gray.shape[0]-1, -1, -1):
        if gray[r, :].any():
            bottom = min(r + 10, gray.shape[0])
            break
    for c in range(gray.shape[1]):
        if gray[:, c].any():
            left = max(c - 10, 0)
            break
    for c in range(gray.shape[1]-1, -1, -1):
        if gray[:, c].any():
            right = min(c + 10, gray.shape[1])
            break
    gray_margin_cut = gray[top:bottom, left:right]
    charac_hei = bottom - top

    # split horizontally
    # histogram of gray is not needed
    split_left = []
    split_right = []
    for idx_sp in range(gray_margin_cut.shape[1]-1):
        if not gray_margin_cut[:, idx_sp].any() and gray_margin_cut[:, idx_sp+1].any():
            # black | white
            split_left.append(idx_sp)
        elif gray_margin_cut[:, idx_sp].any() and not gray_margin_cut[:, idx_sp+1].any():
            # white | black
            split_right.append(idx_sp)
    split_interval = list(zip(split_left, split_right))

    # adapt margin
    characs = []
    for i in split_interval:
        charac = gray_margin_cut[:, i[0]:i[1]]
        dis_hei_wid = (charac_hei - (i[1] - i[0])) // 2
        if dis_hei_wid > (i[1] - i[0]) * 0.1:
            charac = np.pad(charac, ((0, 0), (dis_hei_wid, dis_hei_wid)), 'constant')

        characs.append(
            cv2.threshold(
                cv2.resize(charac, (28, 28), cv2.INTER_CUBIC),
                20, 255, cv2.THRESH_BINARY
            )[1]
        )


    return characs


def main():
    gray = cv2.threshold(
        cv2.imread('./images/formula.jpg', 0), 200, 255, cv2.THRESH_BINARY_INV
    )[1]
    import matplotlib.pyplot as plt
    characs = extract_chars(gray)
    for c in characs:
        plt.imshow(c, cmap='gray')
        plt.show()


if __name__ == '__main__':
    main()
