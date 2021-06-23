import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.colors import hsv_to_rgb
from matplotlib import cm
from matplotlib import colors
import os
from glob import glob
import matplotlib
#matplotlib.use('TkAgg')

def get_image(path, extension, filename_len):
    os.chdir(path)
    print(path)
    image_list = glob('*.{}'.format(extension))
    for file, c in zip(image_list, range(len(image_list))):
        if len(file)<filename_len:
            del(image_list[c])
    return image_list

def plot_cv_img(input_image, output_image3, rename):
    """
    Converts an image from BGR to RGB and plots
    """
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(input_image)
    ax[0].set_title('Input Image')
    ax[0].axis('off')
    #ax[1].imshow(output_image1)
    #ax[1].set_title('Median Filter (5,5)')
    #ax[1].axis('off')
    #ax[2].imshow(output_image2)
    #ax[2].set_title('Median Filter (7,7)')
   # ax[2].axis('off')
    ax[1].imshow(output_image3)
    ax[1].set_title('Median Filter (9,9)')
    ax[1].axis('off')
    plt.savefig(rename)
    plt.show()


path_to_images = 'C:/Users/a-camilleri/Documents/LettuceProject/Dataset_Segmentation/28.05.21 Redmere visit pictures/Field P13 Glassica (Original)/'
#path_to_images = 'C:/Users/a-camilleri/Documents/LettuceProject/Dataset_Segmentation/27.05.21 Redmere visit pictures/Field P12 Glassica/'
image_list = get_image(path_to_images,'png',2)
print(image_list)
for img in image_list:
    image = img
    end = image.find(".")
    mask_rename = 'hsv_mask_stem_seg_' + image
    result_rename = 'hsv_res_stem_seg_' + image
    med_rename = 'med_fil_' + image
    lettuce = cv2.imread(path_to_images+image)
    lettuce = cv2.cvtColor(lettuce, cv2.COLOR_BGR2RGB)
    plt.imshow(lettuce)
    plt.show()

    # compute median filtered image varying kernel size
    #median1 = cv2.medianBlur(lettuce, 5)
    #median2 = cv2.medianBlur(lettuce, 7)
    #median1 = cv2.medianBlur(lettuce, 9)
    # Do plot
    #plot_cv_img(lettuce, median1, med_rename)
    #lettuce = median1
    light_green=(25,20,200)
    dark_green=(50,250,255)
    # lo_square = np.full((5, 5, 3), light_green, dtype=np.uint8) / 255.0
    # do_square = np.full((5, 5, 3), dark_green, dtype=np.uint8) / 255.0
    # plt.subplot(1, 2, 1)
    # plt.imshow(hsv_to_rgb(lo_square))
    # plt.subplot(1, 2, 2)
    # plt.imshow(hsv_to_rgb(do_square))
    # plt.show()

    #Convert RGB to HSV
    hsv_lettuce = cv2.cvtColor(lettuce, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_lettuce)

    mask = cv2.inRange(hsv_lettuce, light_green, dark_green)
    result = cv2.bitwise_and(lettuce, lettuce, mask=mask)
    cv2.imwrite(mask_rename, mask)
    cv2.imwrite(result_rename,cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()

    pixel_colors = lettuce.reshape((np.shape(lettuce)[0]*np.shape(lettuce)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    # fig2 = plt.figure()
    # axis = fig2.add_subplot(1, 1, 1, projection='3d')
    # axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    # axis.set_xlabel("Hue")
    # axis.set_ylabel("Saturation")
    # axis.set_zlabel("Value")
    # axis.view_init(0, 90)
    # plt.show()



