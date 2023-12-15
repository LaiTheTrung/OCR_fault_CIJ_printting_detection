from computorVision_approach.Text_detection import *
import cv2
import time
tool = FirstProgress()
img = cv2.imread('img2.jpg')
while True:
    start = time.time()
    Region, Region_Blur = tool.get_Can_Region(img)
    # print('FPS1: ',1/(time.time()-start+ 10**-6))
    # start = time.time()
    text_regions = tool.GetTextRegion(Region, Region_Blur)
    # print('FPS2: ',1/(time.time()-start+ 10**-6))
    # start = time.time()
    text = tool.TextExtracting(text_regions)
    print('FPS3: ',1/(time.time()-start+ 10**-6))