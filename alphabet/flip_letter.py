import numpy as np
import os
import cv2

folder = './letters/letter_F'
list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
imgs = [cv2.imread(f) for f in list]
print len(imgs), imgs[0].shape

imgs = [img[:,::-1,:] for img in imgs]

cv2.imshow('img', imgs[0])
cv2.waitKey(0)

for i, f in enumerate(list):
    cv2.imwrite(f, imgs[i])

cv2.destroyAllWindows()
