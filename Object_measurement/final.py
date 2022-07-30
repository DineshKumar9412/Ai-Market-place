# 1 Method
# from scipy.spatial import distance as dist
# from imutils import perspective
# # from imutils import contours
# import numpy as np
# import imutils
# import cv2
#
#
# img = cv2.imread('second.jpeg')
#
# # convert to RGB
# image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#
# # create a binary thresholded image
# _, binary = cv2.threshold(gray, 100, 100, cv2.THRESH_BINARY_INV)
# # show i
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# image = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
#
#
# objects_contours = []
# for cnt in contours:
#     if cv2.contourArea(cnt) > 10000:
#         objects_contours.append(cnt)

####################################################################################
#2 Method

import cv2
import numpy as np
from scipy.spatial import distance as dist

filters = 0

main_area = 10000



# def record(my_po):
#     my_new =np.zeros_like(my_po)
#     my_point = my_po.reshape((4, 2))
#     add = my_po.sum(1)
#     my_new[0] = my_point[np.argmin(add)]
#     my_new[3] = my_point[np.argmax(add)]
#     diff = np.diff(my_point, axis=1)
#     my_new[1] = my_point[np.argmin(diff)]
#     my_new[2] = my_point[np.argmax(diff)]
#     return my_new
#
def warpIN(img, point, w, h):
    # print(point.shape)
    pts1 = np.float32(point)

    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    # print(pts2)
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    print(matrix)
    # imgWap = cv2.warpPerspective(img, matrix, (w, h))
    # return imgWap

def mainc():
    image = cv2.imread("one.jpeg")
    image = cv2.resize(image, (0, 0), None, 0.5, 0.5)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 1)
    img_canny = cv2.Canny(img_blur, 100, 100)
    kernal  = np.ones((5, 5))
    img_Dial = cv2.dilate(img_canny, kernal, iterations=3)
    img_thre = cv2.erode(img_Dial, kernal, iterations=2)

    contours, hiearcy = cv2.findContours(img_thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fimalCountours = []

    for i in contours:
        area = cv2.contourArea(i)
        if area > main_area:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            bbox = cv2.boundingRect(approx)
            if filters > 0:
                if len(approx) == filters:
                    fimalCountours.append([len(approx), area, approx, bbox, i])
            else:
                fimalCountours.append([len(approx), area, approx, bbox, i])
    fimalCountours = sorted(fimalCountours, key= lambda x:x[1], reverse=True)

    for con in fimalCountours:
        cv2.drawContours(image, con[4], -1, (0, 0, 225), 3)

    return image, fimalCountours


img, at = mainc()

if len(at) > 0:
    poin = at[0][2]
    # print(poin.shape)
    warpIN(img, poin, 100, 100)


#     cv2.imshow("jij", img)
#     cv2.waitKey(0)
# #
#
# cv2.imshow("Image", img)
# cv2.waitKey(0)












































##################################################################################
# pixel_to_size = None
#
# def mdpt(A, B):
#     return ((A[0] + B[0]) * 0.5, (A[1] + B[1]) * 0.5)
#
# for c in objects_contours:
#     orig = img.copy()
#     bbox = cv2.minAreaRect(c)
#     bbox = cv2.cv.boxPoints(bbox) if imutils.is_cv2() else cv2.boxPoints(bbox)
#     bbox = np.array(bbox, dtype="int") #order the contours and draw bounding box
#     bbox = imutils.perspective.order_points(bbox)
#     cv2.drawContours(orig, [bbox.astype("int")], -1, (0, 255, 0), 2)
#
#     # loop over the original points in bbox and draw them; 5p red dots
#     for (x, y) in bbox:
#         cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
#         # unpack the ordered bounding bbox; find midpoints
#     (tl, tr, br, bl) = bbox
#     (tltrX, tltrY) = mdpt(tl, tr)
#     (blbrX, blbrY) = mdpt(bl, br)
#     (tlblX, tlblY) = mdpt(tl, bl)
#     (trbrX, trbrY) = mdpt(tr, br)
#
#     cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
#     cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
#     cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
#     cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
#     cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (0, 255, 255), 2)
#     cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (0, 255, 255), 2)
#
#     dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
#     dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
#     # print(dA  ,   dB)
#     # use pixel_to_size ratio to compute object size
#     if pixel_to_size is None:
#         pixel_to_size = dB / 0.75
#         distA = dA / pixel_to_size
#         distB = dB / pixel_to_size
#
#         # draw the object sizes on the image
#         cv2.putText(orig, "{:.1f}in".format(distA),
#                     (int(tltrX - 10), int(tltrY - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 2)
#         cv2.putText(orig, "{:.1f}in".format(distB),
#                     (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 2)
#         # show the output image
#         cv2.imshow("Image", orig)
#         cv2.waitKey(0)
