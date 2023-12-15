import cv2
import numpy as np
from scipy.signal import argrelextrema

def GetTheBoundingBox(listBox):
    if len(listBox)>0:
        arr_box= np.array(listBox) 
        x_b = np.min(arr_box[:,0])-3
        y_b = np.min(arr_box[:,1])-3
        w_b = np.max(arr_box[:,0]+arr_box[:,2]) - x_b +3
        h_b = np.max(arr_box[:,1]+arr_box[:,3]) - y_b +3
        return(x_b,y_b,w_b,h_b)
    return False
def RotateRadian(angle_rad,center,size,img):
    #crop have shape(300,300)
    angle_degree = angle_rad*180/np.pi
    M = cv2.getRotationMatrix2D(center, angle_degree, 1.0)
    return cv2.warpAffine(img, M, (size, size))
def RotateDegree(angle_degree,center,size,img):
    #crop have shape(300,300)
    M = cv2.getRotationMatrix2D(center, angle_degree, 1.0)
    return cv2.warpAffine(img, M, (size, size))
def HSV_filter(img):
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = 15
    h_max = 40
    s_min = 0
    s_max = 255
    v_min = 90
    v_max = 255
    lower= np.array([h_min,s_min,v_min])
    upper= np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    #diagnoise
    kernel = np.ones((5, 5), np.uint8)
    new_dilate = cv2.dilate(mask, kernel, iterations=1)
    mask_erosion = cv2.erode(new_dilate, kernel, iterations=2)
    new_mask = cv2.dilate(mask_erosion, kernel, iterations=1)
    imgResult = cv2.bitwise_and(imgHSV,imgHSV,mask = new_mask)
    return imgResult,new_mask

def GetCenter(imgGray):
    filter = np.zeros_like(imgGray)
    m,n = filter.shape
    c = int(m/2)
    filter = cv2.circle(filter, (c,c), int(0.9*c), (255),-1)
    ret,th =  cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    bit_wise0 = np.where(filter==255,th,255 )
    
    circles = cv2.HoughCircles(bit_wise0,cv2.HOUGH_GRADIENT,1.2,100,
                            param1=300,param2=60,minRadius=int(0.5*c),maxRadius=int(0.9*c))
    circles = np.uint16(np.around(circles))[0,:]
    circle_info= np.uint16(np.mean(circles,axis=0))
    return  circle_info


class FirstProgress:
    def __init__(self) -> None:
        self.Input_shape = 640
        self.Crop_shape = 500
        self.Process_shape = 300
        self.TextRegions_ = []
    def get_Can_Region(self,img):
        # the standard of this code is size 1500x1500
        img = cv2.resize(img,(self.Input_shape,self.Input_shape))
        # first filter the color of the image, to get the yellow region
        hsv_filt,new_mask = HSV_filter(img)
        blur = cv2.GaussianBlur(new_mask ,(5,5),0)
        edged = cv2.Canny(blur, 0, 50)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        total_cnt = []
        # diagnoise
        for cnt in contours:
            if cv2.arcLength(cnt, True)>800:
                total_cnt.append(cnt)
        # get the region of can
        total_cnt = np.concatenate(total_cnt,axis = 0)
        box = cv2.boundingRect(total_cnt)
        x,y,w,h = box
        # find the 
        cropImg = img[y:y+h,x:x+w]
        box = cv2.boundingRect(total_cnt)
        cropImg = cv2.resize(cropImg,(self.Crop_shape,self.Crop_shape))
        gray  = cv2.cvtColor(cropImg ,cv2.COLOR_BGR2GRAY)
        cp = GetCenter(gray)
        mask = np.zeros_like(gray)
        cv2.circle(mask, cp[:2], max(cp[2],180), (1),-1)
        Region = cv2.bitwise_and(gray,gray,mask=mask)
        Region_Blur = cv2.GaussianBlur(Region ,(5,5),3)

        return Region, Region_Blur
    
    def GetTextRegion(self,Region,Region_Blur):
        # Initiate FAST object with default values
        fast = cv2.FastFeatureDetector_create()
        fast.setNonmaxSuppression(0)
        # find and draw the keypoints
        kp = fast.detect(Region_Blur,None)
        # diagnoise the outlier points
        filter = np.zeros_like(Region)
        img2 = cv2.drawKeypoints(filter, kp, None, color=255,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)[:,:,0]
        kernel = np.ones((5, 5), np.uint8)
        img2 = cv2.erode(img2, kernel, iterations=1)
        img2_dil = cv2.dilate(img2 , kernel, iterations=1)
        ret,bin2 =  cv2.threshold(img2_dil,128,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(bin2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        main_cnt = []
        for cnt in contours: # get the big contour only
            if cv2.contourArea(cnt) > 1500: 
                main_cnt.append(cnt)
        #get the bounding rectangle of all the big distribution contours
        main_cnt = np.vstack(main_cnt)
        rect =cv2.minAreaRect(main_cnt) # [(x,y),(w,h),angle]
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        angle_r = rect[2] 
        center = [int(rect[0][0]),int(rect[0][1])]
        dim = rect[1]
        # Rotating to horizontal
        if rect[1][0]>rect[1][1]:# width > height
            rotated_ = RotateDegree(img=Region, angle_degree=angle_r, center=center, size=500)
            gray_r = rotated_[int(center[1]-dim[1]/2):int(center[1]+dim[1]/2),
                    int(center[0]-dim[0]/2):int(center[0]+dim[0]/2)]
        else: 
            rotated_ = RotateDegree(img=Region, angle_degree=angle_r-90, center=center, size=500)
            gray_r = rotated_[int(center[1]-dim[0]/2):int(center[1]+dim[0]/2),
                        int(center[0]-dim[1]/2):int(center[0]+dim[1]/2)]
        h,w = gray_r.shape
        height = 200
        width = int(200*w/h)
        gray_r = cv2.resize(gray_r,(width,height)) # result contain 2 line of printing code
        # get the upper and lower line
        # improve the quaility of result
        gamma = 0.4
        gray_c = (np.power((gray_r)/255,gamma)*255).astype(np.uint8)
        thresh2 = cv2.adaptiveThreshold(gray_c, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY_INV,31, 11) 
        # print('FPS1: ',1/(time.time()-start))
        # start = time.time()
        kernel = np.ones((3, 3), np.uint8)

        binary = cv2.dilate(thresh2 , kernel, iterations=1)
        binary = cv2.erode(binary, kernel, iterations=2)
        binary = cv2.dilate(thresh2 , kernel, iterations=1)
        

        edged = cv2.Canny(binary, 100, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        up_cnt = []
        down_cnt =[]
        for cnt in contours:
            if cv2.contourArea(cnt)>70:
                x,y,w,h = cv2.boundingRect(cnt)
                if (h<80):
                    if y+h/2 <100:
                        up_cnt.append(cnt)
                    elif y+h/2 >100:
                        down_cnt.append(cnt)

        up_cnt = np.vstack(up_cnt)
        down_cnt = np.vstack(down_cnt)
        text_regions = []
        boxes_width = []
        pad = 10
            
        for cnt in [up_cnt,down_cnt]:
            rect = cv2.minAreaRect(cnt)
            if rect[1][1]<rect[1][0]:
                rect = (rect[0],(rect[1][1]+pad,rect[1][0]+pad),90+rect[2]) # add padding (x,y),(w,h), angle
            else:
                rect = (rect[0],(rect[1][0]+pad,rect[1][1]+pad),rect[2]) # add padding (x,y),(w,h), angle
            
            boxes_width.append(rect[1][1])
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            height = 80
            width = int(height*rect[1][1]/rect[1][0])

            pts1 = np.float32([box[0],box[1],box[3],box[2]])
            pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
            #wrapping
            matrix = cv2.getPerspectiveTransform(np.array(pts1), pts2)
            text_region = cv2.warpPerspective(binary, matrix, (width , height))
            text_regions.append(text_region)
            text_region = cv2.warpPerspective(gray_r, matrix, (width , height))
            self.TextRegions_.append(text_region)
        # flipping 
        if boxes_width[0] < boxes_width[1]:
            up = cv2.flip(text_regions[1],1)
            up = cv2.flip(up,0)
            down = cv2.flip(text_regions[0],1)
            down = cv2.flip(down ,0)
            text_regions = [up,down]
            up = cv2.flip(self.TextRegions_[1],1)
            up = cv2.flip(up,0)
            down = cv2.flip(self.TextRegions_[0],1)
            down = cv2.flip(down ,0)
            self.TextRegions_ = [up,down]

        return text_regions
    def GetDistribution(self,text_region, lower_line = True):
        # text_region = cv2.bitwise_not(text_region)
        distribute = np.mean(text_region,axis=0)
        filter = np.ones(3)/3
        distribute_conv = np.convolve(filter,distribute)
        f = distribute_conv
        n = len(f)
        fhat = np.fft.fft(f,n)                     # Compute the FFT
        ## Use the PSD to filter out noise
        if not lower_line:
            indices = (np.arange(n)<25)*(np.arange(n)>5)
        else:
            indices = (np.arange(n)<20)*(np.arange(n)>5)
        fhat = indices * fhat     # Zero out small Fourier coeffs. in Y
        ffilt = np.fft.ifft(fhat) # Inverse FFT for filtered time signal
        return ffilt,distribute_conv
    
    def TextExtracting(self,text_regions):
        text = [[],[]]
        
        for i in range(2):
            text_region = text_regions[i]
            ffilt, distribute_conv = self.GetDistribution(text_region, lower_line = i)
            # for local maxima
            list_cen = argrelextrema(ffilt, np.greater)[0] # contain the center of weight point throughout x axis  
            #distribution to get the boundingBox
            power_ffilt = (ffilt-np.min(ffilt))
            rerange_ffilt = power_ffilt/np.max(power_ffilt)
            Super_distribution = rerange_ffilt*(distribute_conv+1)

            change = np.diff(Super_distribution>8)
            list_pos = np.where(change > 0)[0]

            # Contours to get the region of text
            n = len(ffilt)

            kernel = np.ones((4,4))
            thresh_eroded = cv2.dilate(text_region , kernel, iterations=1)
            for boundary in list_pos:
                thresh_eroded[:,boundary-1:boundary+1]=0

            edged = cv2.Canny(thresh_eroded, 100, 200)
            contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            draw = text_region.copy()
            list_boxes = [[] for i in range(len(list_cen))]
            for cnt in contours:
                box = cv2.boundingRect(cnt)
                x,y,w,h= box
                if w*h >100:
                    x,y,w,h = [x+2,y+2,w-2,h-2] 

                    checker1 = (list_cen - x+10)>0
                    checker2 = (x + w - list_cen+10)>0

                    index = np.where((checker1*checker2)==True)[0]
                    if len(index) == 1:
                        list_boxes[index[0]].append([x,y,w,h])
            for list_box in list_boxes:
                box = GetTheBoundingBox(list_box)
                if box:
                    x,y,w,h = box
                    center = int(x+w/2)
                    text_r = self.TextRegions_[i][:,max(center-20,0):min(center+20,n)]
                    # cv2.rectangle(draw,box,color=(255))
                    text[i].append(text_r)

        return text