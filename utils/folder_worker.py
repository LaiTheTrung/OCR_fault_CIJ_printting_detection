import cv2
import os
def renameData(name_set,image_folder ):
    names = [img for img in os.listdir(image_folder)]
    id = 0
    for name in names:
        # try:
        endSwitch = name.split('.')[-1]
        newName = name_set + str(id) + '.' + endSwitch
        os.rename(os.path.join(image_folder,name),os.path.join(image_folder,newName))
        id +=1
def resizeDataSmallerEdgeEqual2Size(file_dir,size):
    list_image = os.listdir(file_dir)
    for name in list_image:
        path = os.path.join(file_dir,name)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        m,n,_ = img.shape
        target = min(m,n)
        scale = size/target
        h =int( m*scale )+1
        w =int( n*scale )+1
        newImage = cv2.resize(img,(w,h),cv2.INTER_CUBIC)
        cv2.imwrite(path,newImage)

def resizeDataBiggerEdgeEqual2Size(file_dir,size):
    list_image = os.listdir(file_dir)
    for name in list_image:
        path = os.path.join(file_dir,name)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        m,n,_ = img.shape
        target = max(m,n)
        scale = size/target
        h =int( m*scale )+1
        w =int( n*scale )+1
        newImage = cv2.resize(img,(w,h),cv2.INTER_CUBIC)
        cv2.imwrite(path,newImage)

def resize2Size(file_dir,size):
    list_image = os.listdir(file_dir)
    for name in list_image:
        path = os.path.join(file_dir,name)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        newImage = cv2.resize(img,size,cv2.INTER_CUBIC)
        cv2.imwrite(path,newImage)