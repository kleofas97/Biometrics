import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('data/101_1.tif', 0)
img2 = cv2.imread('data/101_2.tif', 0)
# img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

# Initiate STAR detector
orb = cv2.ORB_create()

# compute the descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# draw only keypoints location,not size and orientation
img_plot = cv2.drawKeypoints(img1, kp1, outImage=None, color=(0, 255, 0), flags=0)
plt.imshow(img_plot), plt.show()

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img_plot_3= cv2.drawMatches(img1,kp1,img2,kp2,matches[:2], flags=2,outImg=None)

plt.imshow(img_plot_3),plt.show()