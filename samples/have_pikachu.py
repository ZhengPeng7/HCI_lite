import numpy as np
import cv2
from matplotlib import pyplot as plt


img = cv2.imread('pikachu.jpg')
img_ori = img.copy()

mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (300, 10, 950, 700)
cv2.rectangle(img_ori, rect[:2], rect[2:], color=(161, 0, 128), thickness=3)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.figure(figsize=(18,8))
plt.subplot(121)
plt.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
plt.subplot(122)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
