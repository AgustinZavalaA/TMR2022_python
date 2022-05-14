import cv2

img = cv2.imread("src/black_can_img.jpeg")

#set color to image
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#resize just in case and change imshow parammeter
resized_img = cv2.resize(img, (400, 400))
resized_grey_img = cv2.resize(grey_img, (400, 400))
(thresh, resized_bnw) = cv2.threshold(resized_grey_img, 80, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('Normal', resized_img)
cv2.imshow('Gris', resized_grey_img)
cv2.imshow('Black and with with inversal threshold', resized_bnw)

cv2.waitKey()
cv2.destroyAllWindows()














































'''
import cv2
import numpy as np
img = cv2.imread('image.png')

height = img.shape[0]
width = img.shape[1]
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
#cv2.imshow("Mask", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
count = cv2.countNonZero(thresh)
area = count*0.8*0.8/(width*height)
print(area)

img = cv2.imread("image.png")
th, im_th = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY) 

print(th)

cv2.imwrite("image.png", im_th)
'''