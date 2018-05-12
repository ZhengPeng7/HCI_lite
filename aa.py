import cv2
import numpy as np
import time

# get captures
cap = cv2.VideoCapture(0)
SIZE = (200, 150)
counter = -1
while cap.isOpened():
    counter += 1
    start_time_extract_figure = time.time()
    # extract your figure
    _, frame = cap.read()
    frame = cv2.resize(frame, SIZE, interpolation=cv2.INTER_CUBIC)
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (2, 2, SIZE[1] - 10, SIZE[0] - 10)
    start_time_grabCut = time.time()
    cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), (0,), (1,)).astype('uint8')
    frame = frame * mask2[:, :, np.newaxis]

    elapsed_time_extract_figure = time.time() - start_time_extract_figure
    # print('{}-th extract_figure_time: {}'.format(counter, elapsed_time_extract_figure))
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_CUBIC)
    cv2.putText(frame, 'fps: '+str(1//elapsed_time_extract_figure), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), thickness=2)
    cv2.imshow(
        'frame',
        frame
    )

    k = cv2.waitKey(3) & 0xff
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()