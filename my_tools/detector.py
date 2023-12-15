import cv2
import numpy as np
from paddleocr import PaddleOCR,check_img
from paddleocr.tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop
class First_progress:
    def __init__(self,Input_shape = 1500, Crop_shape = 500, Process_shape = 300) -> None:
        self.Input_shape = Input_shape
        self.Crop_shape = Crop_shape
        self.Process_shape =Process_shape
        self.a = int(110/300*self.Process_shape)
        self.b = int(190/300*self.Process_shape)
        self.Half_shape = self.Crop_shape/2
        self.box =[]
        self.detector = PaddleOCR( use_angle_cls=False)
    # basic subset function for image processing


    def Diagnoise(self,kernel_size,erosion_iter,dilate_iter,img, erosion_first = True):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if erosion_first:
            erosion = cv2.erode(img, kernel, iterations=erosion_iter)
            return cv2.dilate(erosion, kernel, iterations=dilate_iter)
        else:
            dilate = cv2.dilate(img, kernel, iterations=dilate_iter)
            return cv2.erode(dilate, kernel, iterations=erosion_iter)
        
        
    def hsv_equalizedV2(self,BGRimage,CLAHE = False ):
        if not CLAHE:
            H, S, V = cv2.split(cv2.cvtColor(BGRimage, cv2.COLOR_BGR2HSV))
            eq_S = cv2.equalizeHist(S)
            eq_image = cv2.cvtColor(cv2.merge([H, eq_S, V]), cv2.COLOR_HSV2RGB)
        else:
            lab= cv2.cvtColor(BGRimage, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl,a,b))
            eq_image =cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return eq_image
    

    def hsv_mask_creator(self,h_range,s_range,v_range,imgHSV):
        h_min = h_range[0]
        h_max = h_range[1]
        s_min = s_range[0]
        s_max = s_range[1]
        v_min = v_range[0]
        v_max = v_range[1]
        lower= np.array([h_min,s_min,v_min])
        upper= np.array([h_max,s_max,v_max])
        mask = cv2.inRange(imgHSV,lower,upper)
        return mask
    

    def RotateRadian(self,angle_rad,size,crop):
        #crop have shape(300,300)
        angle_degree = angle_rad*180/np.pi
        M = cv2.getRotationMatrix2D((size/2,size/2), angle_degree, 1.0)
        return cv2.warpAffine(crop, M, (size, size))
    

    def get_Loss(self,rotated_img):
        check_Region = rotated_img[110:190,110:190]
        gray = cv2.cvtColor(check_Region,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray ,(9,9),0)
        L2 = np.mean(np.mean(255-gray,axis=1)*(np.arange(110,190)-150)**2)
        return L2
    
    def text_detector(self, detector,img):
        dt_boxes = detector.ocr(img,det=True,rec=False,cls=False)
        if dt_boxes is None:
            return None, None
        return dt_boxes
    
    def get_new_region(self,posA,posB):
        xA,yA = posA
        xB,yB = posB
        return np.array([min(xA,xB),max(yA,yB)])
    # application function 


    def get_Can_Region(self,imgHSV):
        imgHSV = cv2.resize(imgHSV,(self.Input_shape,self.Input_shape))
        mask = self.hsv_mask_creator(h_range=[14,40],s_range=[15,255],v_range=[50,255],imgHSV=imgHSV)
        new_mask = self.Diagnoise(8,erosion_iter=2,dilate_iter=7,img=mask)
        imgResult = cv2.bitwise_and(imgHSV,imgHSV,mask = new_mask)
        blur = cv2.GaussianBlur(new_mask ,(5,5),0)
        edged = cv2.Canny(blur, 0, 50)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        total_cnt = []
        for cnt in contours:
            if cv2.arcLength(cnt, True)>100:
                total_cnt.append(cnt)
        total_cnt = np.concatenate(total_cnt,axis = 0)
        x,y,w,h = cv2.boundingRect(total_cnt)
        cropImg = imgResult[y:y+h,x:x+w]
        cropImg = cv2.resize(cropImg,(self.Crop_shape,self.Crop_shape))
        cropImg  = cv2.cvtColor(cropImg ,cv2.COLOR_HSV2BGR)
        return [x,y,w,h],cropImg
    

    def RotateCrop(self, cropImg):
        Resize_crop = cv2.resize(cropImg,(self.Process_shape,self.Process_shape)) 
        print(Resize_crop.shape)   
        check_Region = Resize_crop[self.a:self.b,self.a:self.b]

        gray = cv2.cvtColor(check_Region,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray ,(5,5),0)
        ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        lines = cv2.HoughLinesP(th, 1, np.pi/180, 10, minLineLength=50, maxLineGap=2)

        theta_arr = np.array([0])
        for line in lines:
            for x1, y1, x2, y2 in line:
                theta = np.arctan2(y2-y1,x2-x1)
                checker = theta_arr-theta
                if np.min(np.abs(checker)) < 0.12:
                    continue
                else:
                    theta_arr = np.append(theta_arr,theta)

        min_loss = 0
        result = 0
        for theta in theta_arr:
            rotated = self.RotateRadian(theta,300,Resize_crop) 
            loss = self.get_Loss(rotated)
            if min_loss == 0:
                min_loss = loss
                result = theta
            if loss < min_loss:
                min_loss == loss
                result = theta
        
        return result,self.RotateRadian(result,500, cropImg)
    
    def GetTextRegion(self, cropImg, AI_approach = True):
        if AI_approach:
            boxes = self.text_detector(self.detector,cropImg)
            container = []
            # warp perspective to the rectangle size
            for box in boxes[0]:
                p1 = np.array(box[0])
                p2 = np.array(box[1])
                p3 = np.array(box[3])
                p4 = np.array(box[2])
                h = int(np.sqrt(np.sum(np.power(p3-p1,2))))
                w = int(np.sqrt(np.sum(np.power(p2-p1,2))))
                height = 60
                width = int(60*w/h)
                pts1 = np.float32([p1,p2,p3,p4])
                pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                result = cv2.warpPerspective(cropImg, matrix, (width , height))
                container.append(result)
            return container
        else:
            h = int(self.Crop_shape/5)
            w = int(self.Crop_shape*330/500)
            test = cropImg[int(self.Half_shape-h/2):int(self.Half_shape+h/2), int(self.Half_shape-w/2):int(self.Half_shape+w/2),:]
            print(test.shape)
            test = cv2.resize(test,[w*2,h*2])
            gray_c = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
            thresh2 = cv2.adaptiveThreshold(gray_c, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                    cv2.THRESH_BINARY,19, 11) 
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.erode(thresh2, kernel, iterations=1)
            binary = cv2.medianBlur(binary,13)
            binary = cv2.dilate(binary , kernel, iterations=1)

            # plt.imshow(th)
            blur = cv2.GaussianBlur(binary ,(3,3),0)
            edged = cv2.Canny(blur, 100, 200)
            contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
 
            up_cnt = []
            down_cnt =[]
            boxes = []
            for cnt in contours:
                if cv2.contourArea(cnt)>50:
                    x,y,w_s,h_s = cv2.boundingRect(cnt)
                    if y+h_s/2 <h/2:
                        up_cnt.append(cnt)
                    elif y+h_s/2 >h/2:
                        down_cnt.append(cnt)
            up_cnt = np.vstack(up_cnt)
            down_cnt = np.vstack(down_cnt)
            for cnt in [up_cnt,down_cnt]:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                boxes.append(box)
            container = []
            print(boxes)
            pad = 6
            for point in boxes:
                p1 = np.array([point[1][0]-pad,point[1][1]-pad])
                p2 = np.array([point[2][0]+pad,point[2][1]-pad])
                p3 = np.array([point[0][0]-pad,point[0][1]+pad])
                p4 = np.array([point[3][0]+pad,point[3][1]+pad])
                h = int(np.sqrt(np.sum(np.power(p3-p1,2))))
                w = int(np.sqrt(np.sum(np.power(p2-p1,2))))
                height = 60
                width = int(60*w/h)
                pts1 = np.float32([p1,p2,p3,p4])
                pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])    
                matrix = cv2.getPerspectiveTransform(np.array(pts1), pts2)
                result = cv2.warpPerspective(test, matrix, (width , height))
                container.append(result)
            return container

    def TextExtractorTool(self,container):
        text =[]
        for line in container:
            #preprocessing
            gray_c = cv2.cvtColor(line,cv2.COLOR_BGR2GRAY)

            thresh2 = cv2.adaptiveThreshold(gray_c, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 19, 11)
            thresh2[0:2,:] = 255
            thresh2[-2:,:] = 255
            thresh2 = np.pad(thresh2,2,constant_values=255)
            # get the region of each word
            kernel = np.ones((3,3))
            thresh_eroded = cv2.erode(thresh2, kernel, iterations=1)
            blur = cv2.GaussianBlur(thresh_eroded ,(3,3),0)
            edged = cv2.Canny(blur, 100, 200)
            contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            list_pos = np.array([[0,0]])
            count = 0
            for cnt in contours:
                box = cv2.boundingRect(cnt)
                x,y,w,h= box
                if (w*h>250):
                    count +=1
                    pos = np.array([x,x+w])
                    center = x + w/2
                    checker = list_pos - center
                    checker = (checker[:,0]*checker[:,1])<0    
                    if checker.any()==True:
                        idx = np.where(checker==True)[0][0]
                        list_pos[idx] = self.get_new_region(list_pos[idx],pos)
                    else:
                        list_pos = np.vstack([list_pos,pos])
            list_pos = np.sort(list_pos,axis=0)
            for pos in list_pos[1:]:
                start = pos[0]
                stop = pos[1]
                text.append(line[:,start:stop,:])
        return text
    
    def GetText(self,cropImage,AI_approach):
        container = self.GetTextRegion(cropImage, AI_approach = AI_approach)
        list_text = self.TextExtractorTool(container)
        return list_text
    
    
