import cv2
import os
import numpy as np
def remove_folder(folder_path):
    if os.path.exists(folder_path):
        filePath = folder_path
        if len(os.listdir(filePath)) > 0:
            for name in os.listdir(filePath):
                os.remove(filePath+r'/'+name)
        os.removedirs(folder_path)
def renameImgInPath(image_folder,name_img, endswith):# try to change name frame1.png, frame11.png, frame111.png to frame001.png,frame011.png,frame111.png
    names = [img for img in os.listdir(image_folder) if img.endswith(endswith)]
    length = len(names)
    id_start = len(name_img)
    id_end = -len(endswith)
    zero_format = int(np.floor(np.log10(length))+1) # get the number of zeros for the 0th index name
    for name in names:
        # try:
        id = int(name[id_start:id_end])
        newName = name_img + str(id).zfill(zero_format)+endswith
        os.rename(os.path.join(image_folder,name),os.path.join(image_folder,newName))
        # except Exception as e:
        #     print(e)
def Images2Video(video_path, image_folder, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    print(height,width,layers)
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    for image in images:
        name = os.path.join(image_folder, image)
        print(name)
        video.write(cv2.imread(name))

    cv2.destroyAllWindows()
    video.release()
def Video2Images(video_path, image_folder,fps,endswitch):
    remove_folder(image_folder)
    os.makedirs(image_folder)
    count = 0
    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(fps)) 
    success,image = vidcap.read()
    success = True
    while success:
        success,image = vidcap.read()   # added this line 
        if success:
            print ('Read a new frame: ', success)
            cv2.imwrite( image_folder + "\\frame%d" % count + endswitch, image)     # save frame as JPEG file
            count = count + 1
    vidcap.release()