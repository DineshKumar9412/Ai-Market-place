import cv2
from object_detection import *
import numpy as np


img = cv2.imread("second.jpeg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

objects_contours = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    # print(area)
    if area > 5000:
        objects_contours.append(cnt)

for cnts in objects_contours:
    rect = cv2.minAreaRect(cnts)
    (x, y), (w, h), angle = rect
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)
    cv2.polylines(img, [box], True, (255, 0, 0), 2)
    cv2.putText(img, "Width {} cm".format(round(w, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.putText(img, "Height {} cm".format(round(h, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


cv2.imshow("Image", img)
cv2.waitKey(0)


# countors = decorator.detect_object(img)

# cv2.imshow("hgh", mask)
# cv2.waitKey(0)