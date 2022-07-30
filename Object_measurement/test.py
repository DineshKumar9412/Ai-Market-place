import cv2
import numpy as np
# from imutils import perspective
# from imutils import contours
import imutils

image = cv2.imread('second.jpeg')

# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# create a binary thresholded image
_, binary = cv2.threshold(gray, 100, 100, cv2.THRESH_BINARY_INV)
# show i
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

objects_contours = []
for cnt in contours:
    if cv2.contourArea(cnt) > 5000:
        objects_contours.append(cnt)






# # Prepocess
# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray,(1,1),1000)
# flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
#
# # Find contours
# contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# contours = sorted(contours, key=cv2.contourArea,reverse=True)
#
# # Select long perimeters only
# perimeters = [cv2.arcLength(contours[i],True) for i in range(len(contours))]
# listindex=[i for i in range(15) if perimeters[i]>perimeters[0]/2]
# numcards=len(listindex)
#
# imgcont = image.copy()
# [cv2.drawContours(imgcont, [contours[i]], 0, (0,255,0), 5) for i in listindex]


# imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(imgray, 127, 255, 0)
# contours, img2 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# # cnt = contours[4]
# # cv2.drawContours(image, [cnt], 0, (0,255,0), 3)
# cv2.drawContours(image, contours, -1, (0,255,0), 3)
# cv2.drawContours(image, contours, 3, (0,255,0), 3)
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# #
# g = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
# edge = cv2.Canny(g, 60, 180)
#
# contours = cv2.findContours(edge,
#                             cv2.RETR_EXTERNAL,
#                             cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(image, contours[0], -1, (0,0,255), thickness = 2)

# contours, h = cv2.findContours(edge,
#                                cv2.RETR_EXTERNAL,
#                                cv2.CHAIN_APPROX_NONE)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)
# cv2.drawContours(image, contours[0], -1, (0,0,255), thickness = 5)

# gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# r, t = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
# contours, h = cv2.findContours(t, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)
# cv2.drawContours(image, contours, -1, (0,0,255), thickness = 5)


# g = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
# edge = cv2.Canny(g, 100, 100)
# # contours, hierarchy = cv2.findContours(edge,
# #     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# contours, hierarchy = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# for c in contours:
#     hull = cv2.convexHull(c)
#     cv2.drawContours(image, [hull], 0, (0, 255, 0), 2)

