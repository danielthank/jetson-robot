from __future__ import print_function
from letter_detector import FindLetter
from keras.models import load_model
import cv2
import numpy as np
import os

def calcHist(img, bins):
    hist = np.zeros(bins)
    for row in img:
        for pix in row:
            hist[pix] += 1
    return hist

COLOR_MAP = {0 : "RED",
             1 : "ORANGE",
             2 : "YELLOW",
             3 : "BLUE",
             4 : "GRASS",
             5 : "GREEN",
             6 : "PINK",
             7 : "SKY"}

## load images ##
LETTER_MAP = {0 : "A",
              1 : "B",
              2 : "C",
              3 : "D",
              4 : "G",
              5 : "J",
              6 : "K",
              7 : "L",
              8 : "M",
              9 : "N",
              10 : "O",
              11 : "P",
              12 : "R",
              13 : "S",
              14 : "U",
              15 : "V",
              16 : "W",
              17 : "Z"}

model_path = 'letter_model.h5'

if os.path.isfile(model_path):
    model = load_model(model_path)
else:
    print('no pre-trained exist!')
    exit(0)

test_img = cv2.imread('./test1.jpg')
test_img = cv2.resize(test_img, (640, 480))
img_letters, cnts_merge_f, cnts_fit, cnts_minRect_orig, cnts_minRect, is_blocks = FindLetter(test_img, show_result=False)
if not cnts_merge_f == None:
    for i, rect in enumerate(cnts_minRect_orig):
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(test_img ,[box], 0, (0,0,255), 2)

        img_letter = cv2.resize(img_letters[i], (32, 32))
        hsv = cv2.cvtColor(img_letter, cv2.COLOR_BGR2HSV)
        hue = hsv[:,:,0]
        sat = hsv[:,:,1]
        value = hsv[:,:,2]
        hist = np.concatenate([calcHist(hue, 180), calcHist(sat, 256), calcHist(value, 256)])
        img_letter = np.expand_dims(np.array(img_letter, dtype='float32').transpose(2, 0, 1)/255., axis=0)
        hist = np.expand_dims(np.array(hist, dtype='float32')/(32*32), axis=0)

        letter_pred, color_pred = model.predict_on_batch([img_letter, hist])
        #print(LETTER_MAP[letter_pred.argmax()], COLOR_MAP[color_pred.argmax()])
        #cv2.imshow('img', img_letters[i])
        #cv2.waitKey(0)
        cv2.putText(test_img, LETTER_MAP[letter_pred.argmax()]+' '+COLOR_MAP[color_pred.argmax()], tuple(np.asarray(cnts_minRect_orig[i][0], dtype='int')), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), thickness=1)

cv2.imshow('img', test_img)
cv2.waitKey(0)

img_letter = cv2.imread('./letters/letter_W/W1003.jpg')
img = cv2.resize(img_letter, (32, 32))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue = hsv[:,:,0]
sat = hsv[:,:,1]
value = hsv[:,:,2]
hist = np.concatenate([calcHist(hue, 180), calcHist(sat, 256), calcHist(value, 256)])
img = np.expand_dims(np.array(img, dtype='float32').transpose(2, 0, 1)/255., axis=0)
hist = np.expand_dims(np.array(hist, dtype='float32')/(32*32), axis=0)

letter_pred, color_pred = model.predict_on_batch([img, hist])
print(LETTER_MAP[letter_pred.argmax()], COLOR_MAP[color_pred.argmax()])
cv2.imshow('img', img_letter)
cv2.waitKey(0)

cv2.destroyAllWindows()
