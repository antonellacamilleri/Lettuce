import cv2

path_to_image0 = 'C:/Users/a-camilleri/Documents/LettuceProject/Dataset_Segmentation/27.05.21 Redmere visit pictures/FieldP13_Carbine_Seg_Med_HSV/hsv_res_seg_4-01.jpg'
path_to_image1 = 'C:/Users/a-camilleri/Documents/LettuceProject/Dataset_Segmentation/27.05.21 Redmere visit pictures/FieldP13_Carbine_Seg_Med_HSV/hsv_res_seg_5-01.jpg'

# read the images
img1 = cv2.imread(path_to_image0)
img2 = cv2.imread(path_to_image1)

# convert images to grayscale
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# create SIFT object
sift = cv2.xfeatures2d.SIFT_create()

# detect SIFT features in both images
keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)


# create feature matcher
#NORM_L2, NORM_HAMMING, NORM_HAMMING2.
bf = cv2.BFMatcher(cv2.NORM_L2SQR, crossCheck=True)
# match descriptors of both images
matches = bf.match(descriptors_1,descriptors_2)

# sort matches by distance
matches = sorted(matches, key = lambda x:x.distance)
# draw first 50 matches
matched_img = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[0:500], img2, flags=2)

# show the image
cv2.imshow('image', matched_img)
# save the image
cv2.imwrite("matched_images.jpg", matched_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()