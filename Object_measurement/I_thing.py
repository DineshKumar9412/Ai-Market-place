from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2


# define a video capture object
vid = cv2.VideoCapture(1)

pixel_to_size = None

def mdpt(A, B):
    return ((A[0] + B[0]) * 0.5, (A[1] + B[1]) * 0.5)

while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edge_detect = cv2.Canny(gray, 80, 200)  # play w/min and maxvalues to finetune edges (2nd n 3rd params)
    edge_detect = cv2.dilate(edge_detect, None, iterations=3)
    edge_detect = cv2.erode(edge_detect, None, iterations=2)

    cntours = cv2.findContours(edge_detect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cntours[0] == ():
        continue
    cntours = imutils.grab_contours(cntours)
    (cntours, _) = contours.sort_contours(cntours)

    for c in cntours:
        if cv2.contourArea(c) < 10000:
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        bbox = perspective.order_points(box)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

        for (x, y) in bbox:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        (tl, tr, br, bl) = bbox
        (tltrX, tltrY) = mdpt(tl, tr)
        (blbrX, blbrY) = mdpt(bl, br)
        (tlblX, tlblY) = mdpt(tl, bl)
        (trbrX, trbrY) = mdpt(tr, br)

        cv2.circle(frame, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        cv2.line(frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (0, 255, 255), 2)
        cv2.line(frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (0, 255, 255), 2)

        distA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        distB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))


        # use pixel_to_size ratio to compute object size
        #
        # pixel_to_size = dB / 0.75
        # distA = dA / pixel_to_size
        # distB = dB / pixel_to_size

        # draw the object sizes on the image
        cv2.putText(frame, "{:.1f}cm".format(distA),
                    (int(tltrX - 10), int(tltrY - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, "{:.1f}cm".format(distB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

