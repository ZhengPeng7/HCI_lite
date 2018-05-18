import cv2
import numpy as np
import face_recognition


def wear_glasses(frame, args_glass):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    centers = []

    face_locations = face_recognition.face_locations(frame)
    for (top, right, bottom, left) in face_locations:
        roi_gray = gray[top:bottom, left:right]
        eyes = args_glass['eye_cascade'].detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            centers.append((left + int(ex + 0.5 * ew), top + int(ey + 0.5 * eh)))

    if len(centers) == 2:
        glasses_width = 2.16 * abs(centers[1][0] - centers[0][0])
        overlay_img = np.ones(frame.shape, np.uint8) * 255
        h, w = args_glass['glass_img'].shape[:2]
        scaling_factor = glasses_width / w

        overlay_glasses = cv2.resize(args_glass['glass_img'], None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        x = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0]

        x -= 0.26 * overlay_glasses.shape[1]
        y = top
        y += 0.55 * overlay_glasses.shape[0]

        h, w = overlay_glasses.shape[:2]
        overlay_img[int(y):int(y + h), int(x):int(x + w)] = overlay_glasses

        # Create a mask and generate it's inverse.
        gray_glasses = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray_glasses, 110, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        temp = cv2.bitwise_and(frame, frame, mask=mask)

        temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
        final_img = cv2.add(temp, temp2)

    else:
        final_img = frame
    
    return final_img
