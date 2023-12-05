import os
import cv2
import numpy as np
from PIL import Image

from skimage import io
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from matplotlib.figure import Figure

#参数配置
S = 60
black = np.array([0, 0, 0])
red = 55
green = 55
blue = 55
k=30

im = cv2.imread('test11.jpg')

imshow(im)
# im = cv2.GaussianBlur(im, (3, 3), 0)
# Convert original image to grayscale
grayscale_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# cv2.imshow("Gray", grayscale_im)
imshow(grayscale_im)
# Apply binary tresholding to the image
# _, im_bin = cv2.threshold(grayscale_im, 0, 255,
#                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)


im_bin = cv2.adaptiveThreshold(grayscale_im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 5, 5)

im_bin = 255 - im_bin
imshow(im_bin)


# im_bin= morphology.thin(im_bin)

# Fill horizontal gaps, 8 pixels wide
selem_horizontal = morphology.rectangle(2, 8)
im_filtered = morphology.closing(im_bin, selem_horizontal)

# Fill vertical gaps, 8 pixels wide
selem_vertical = morphology.rectangle(8, 2)
im_filtered = morphology.closing(im_bin, selem_vertical)


im_filtered = im_filtered.astype(np.uint8)
plt.imshow(im_filtered)

# Kernel used to detect all the horisontal lines
kernel_h = np.ones((20, 1), np.uint8)

# Kernel to detect all the vertical lines
kernel_v = np.ones((1, 20), np.uint8)

# Horizontal kernel on the image
im_bin_h = cv2.morphologyEx(im_filtered, cv2.MORPH_OPEN, kernel_h)
# Image.fromarray(img_bin_h).show()
Figure()
plt.imshow(im_bin_h)
# Verical kernel on the image
im_bin_v = cv2.morphologyEx(im_filtered, cv2.MORPH_OPEN, kernel_v)
# Image.fromarray(img_bin_v).show()
Figure()
plt.imshow(im_bin_v)
# Combining the image
im_final = im_bin_h | im_bin_v
# Apply dilation on our image, to fill potential small gaps
dilation_kernel = np.ones((1, 7), np.uint8)
im_dilated = cv2.dilate(im_final, dilation_kernel, iterations=1)

dilation_kernel = np.ones((7, 1), np.uint8)
im_dilated = cv2.dilate(im_dilated, dilation_kernel, iterations=1)

im_final = im_dilated
Figure()
imshow(im_final)

imlabel = morphology.label(im_final)
ar3 = imlabel > 0
c = morphology.remove_small_objects(ar3, 5000, connectivity=8)
Figure()
imshow(c)

[m, n] = np.shape(im_final)


D = np.zeros((m, n), dtype='uint8')
for i in range(m):
    for j in range(n):
        if c[i, j] == True:
            D[i, j] = 0
        else:
            D[i, j] = im_final[i, j]
Image.fromarray(D).show()


# Find all connected components
_, labels, stats, _ = cv2.connectedComponentsWithStats(
    ~D, connectivity=4, ltype=cv2.CV_32S)

for x, y, w, h, area in stats[2:]:

    # Values to filter components with
    keepWidth = w >= (S-40) and w <= S
    keepHeight = h >= (S-40) and h <= S
    keepProportion = w/h <= 3/2 + 0.2 and h <= w + 7

    if all((keepWidth, keepHeight, keepProportion)):

        # Check the relation between black and white pixels
        # inside the component
        blackPx = 0
        whitePx = 0
        for i in range(x, x+w-1):
            for j in range(y, y+h-1):
                if im[j, i][0] < red and im[j, i][1] < green and im[j, i][2] < blue:
                    blackPx += 1
                else:
                    whitePx += 1                                         # 将像素标记为蓝色
            print(f"Pixel at ({i}, {j}): {im[j, i]}")        
        print(f"black:{blackPx}")
        print(f"white:{whitePx}")
        # Condition for checked box
        if k*blackPx < whitePx:
            cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
Image.fromarray(im).show()
