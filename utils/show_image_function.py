import cv2
import matplotlib.pyplot as plt

def show_images(img_list,name_list = None, max_column = 5,cmap = None):
    if len(img_list) == 0:
        return
    else:
        try:
            n = len(img_list)
            row = n // max_column+1
            fig,a =  plt.subplots(row,max_column)
            if row == 1:
                a = a.reshape(1,max_column)
            if max_column == 1:
                a = a.reshape(row,1)
            if name_list == None:
                for i in range(n):
                    row_plot = i // max_column
                    col_plot = i % max_column
                    print(row_plot,col_plot)
                    a[row_plot][col_plot].imshow(img_list[i])
            else:
                for i in range(n):
                    row_plot = i // max_column
                    col_plot = i % max_column
                    a[row_plot][col_plot].imshow(img_list[i])
                    a[row_plot][col_plot].set_title(name_list[i])
            
            if isinstance(cmap,None):
                plt.imshow()
            else:
                plt.imshow(cmap=cmap)
        except Exception as e:
            print(e)