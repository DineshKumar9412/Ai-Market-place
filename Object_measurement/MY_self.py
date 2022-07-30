from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2


# define a video capture object
vid = cv2.VideoCapture(0)

pixel_to_size = None

def mdpt(A, B):
    return ((A[0] + B[0]) * 0.5, (A[1] + B[1]) * 0.5)


while True:
    ret, frame = vid.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    #
    # cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edge_detect = cv2.Canny(gray, 80, 200)  # play w/min and maxvalues to finetune edges (2nd n 3rd params)
    edge_detect = cv2.dilate(edge_detect, None, iterations=3)
    edge_detect = cv2.erode(edge_detect, None, iterations=2)

    cntours = cv2.findContours(edge_detect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntours = imutils.grab_contours(cntours)
    (cntours, _) = contours.sort_contours(cntours)

    for c in cntours:
        if cv2.contourArea(c) < 20000:
            continue


        (x, y, w, h) = cv2.boundingRect(c)
        print(x )
        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

#         M = cv2.moments(c)
#         print(M)
# #         cx = int(M['m10'] / M['m00'])
# #         cy = int(M['m01'] / M['m00'])
#
#
#         # print(x, y, w, h)
#
#         # cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
#         cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
# #
#         cv2.rectangle(frame, (x, y), (x + w , y + h), (0, 0, 255))
#         cv2.putText(frame, str(cv2.contourArea(c)), (x, y), 0, 1, (0, 255, 0))
#
#
#
#         # cv2.rectangle(frame, (x, y), (x + w , y + h), (0, 0, 255))

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

    ##############################################
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # edge_detect = cv2.Canny(gray, 80, 200)  # play w/min and maxvalues to finetune edges (2nd n 3rd params)
    # edge_detect = cv2.dilate(edge_detect, None, iterations=3)
    # edge_detect = cv2.erode(edge_detect, None, iterations=2)
    #
    # cntours = cv2.findContours(edge_detect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # print(cntours)
    #
    # cntours = imutils.grab_contours(cntours)
    #
    # (cntours, _) = contours.sort_contours(cntours)

 ########################################################
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
    #
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for c in cntours:
#         area = cv2.contourArea(c)
#         if area > 2000:
#             (x, y, w, h) = cv2.boundingRect(c)
#             cv2.putText(frame, str(area), (x, y), 0, 1, (0, 255, 0))
#             cv2.rectangle(frame, (x, y), (x + w , y + h), (0, 0, 255))
#
#     cv2.imshow('frame', edge_detect)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#
#
# vid.release()
# cv2.destroyAllWindows()
#
# one #

# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (3, 3), 0)
# edge_detect = cv2.Canny(gray, 80, 200)  # play w/min and maxvalues to finetune edges (2nd n 3rd params)
# edge_detect = cv2.dilate(edge_detect, None, iterations=3)
# edge_detect = cv2.erode(edge_detect, None, iterations=2)
#
# cntours = cv2.findContours(edge_detect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # print(cntours)
#
# cntours = imutils.grab_contours(cntours)
#
# (cntours, _) = contours.sort_contours(cntours)


# Two ##

# frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# frame2 = cv2.GaussianBlur(frame1,(15,15),0)
#
# frame4 = cv2.threshold(frame2, 15, 255, cv2.THRESH_BINARY)[1]
# kernel = np.ones((2, 2), np.uint8)
# frame5 = cv2.erode(frame4, kernel, iterations=4)
# frame5 = cv2.dilate(frame5, kernel, iterations=8)
# contours, nada = cv2.findContours(frame5.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
