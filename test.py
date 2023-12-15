from my_tools.detector import First_progress
import numpy as np
import cv2
tool = First_progress()
img = cv2.imread('img2.jpg')
imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
box,cropImg =  tool.get_Can_Region(imgHSV)
angle, rotated = tool.RotateCrop(cropImg)
# angle, rotated = processor.RotateCrop(rotated)
text = tool.GetText(rotated,AI_approach=True)
print(angle)
cv2.imshow('original',cv2.resize(img,(500,500)))
cv2.imshow('crop',crop)
cv2.imshow('after_process',rotated)
list_t = []
for t in text:
    t = cv2.resize(t,(70,140))
    t = cv2.cvtColor(t,cv2.COLOR_BGR2GRAY)
    t = np.pad(t,2)
#     print('aaaaaaaaaaa',t.shape)
    list_t.append(t)
# print (len(list_t))
show_down = np.hstack(list_t[:15])
# show_up = np.hstack(list_t[15:])
# cv2.imshow('text_up', show_up)
cv2.imshow('text_down', show_down)

cv2.waitKey(0)