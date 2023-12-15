import cv2 as cv
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle
import matplotlib.pyplot as plt
# Load MLP model for digit classifier
mlp = pickle.load(open('mlpModel.sav','rb'))

def digit_classifier(crop):
    # padding 0
    # top,bottom padding
    h = crop.shape[0]; w = crop.shape[1]
    row_padding = np.zeros((int(h*0.2),w), dtype=np.uint8) 
    padded_roi = np.vstack((row_padding,crop,row_padding)) 
    # left,right padding
    new_h = padded_roi.shape[0]
    col_padding = np.zeros((new_h,int((new_h-w)/2)), dtype=np.uint8)
    padded_roi = np.hstack((col_padding,padded_roi,col_padding)) 
    # cv.imshow('pad',padded_roi)
    # cv.waitKey(0)
    # resize img
    padded_roi = cv.resize(padded_roi,(28,28))
    # normalize
    padded_roi = padded_roi.astype(np.float32)
    cv.normalize(padded_roi, padded_roi, 0,1,cv.NORM_MINMAX)
    # Flatten
    padded_roi = padded_roi.reshape((1,-1))
    # predict
    y_predict = mlp.predict(padded_roi)[0]

    return int(y_predict)

# load test image
img = cv.imread('number2.jpg', cv.IMREAD_REDUCED_COLOR_2)
# Convert color image to gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5,5), 0)

# binarize the image
thresh = 200
b_img = cv.threshold(gray,thresh,255,cv.THRESH_BINARY_INV)[1]
# find object contours
contours, hierachy = cv.findContours(b_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for index,cnt in enumerate(contours):
    x,y,w,h = cv.boundingRect(cnt)
    if h>10:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi = b_img[y:y+h, x:x+w] 
        y_pred = digit_classifier(roi)   
        text = str(y_pred)
        cv.putText(img, text,(x,y), cv.FONT_HERSHEY_COMPLEX,1,(255,0,255),1)

cv.imshow('raw',img)
# cv.imshow('image',b_img)
cv.waitKey(0)
cv.destroyAllWindows()