import cv2
import numpy as np

# wabcam = False
img_path = "third.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img, (0, 0), None, 0.5, 0.5)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5,5), 1)
img_canny = cv2.Canny(img_blur, 100, 100)
kernal  = np.ones((5, 5))
img_Dial = cv2.dilate(img_canny, kernal, iterations=3)
img_thre = cv2.erode(img_Dial, kernal, iterations=2)

contours, hiearcy = cv2.findContours(img_thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
fimalCountours = []
for i in contours:
    area = cv2.contourArea(i)
    if area > 1000:
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02*peri, True)
        bbox = cv2.boundingRect(approx)
        filters = 0
        if filters > 0:
            if len(approx) == filters:
                fimalCountours.append([len(approx), area, approx, bbox, i])
        else:
            fimalCountours.append([len(approx), area, approx, bbox, i])
fimalCountours = sorted(fimalCountours, key= lambda x:x[1], reverse=True)
#
for con in fimalCountours:
    cv2.drawContours(img, con[4], -1, (0, 0, 225), 3)
#     if len(con) != 0:
#         big = con[0][2]


    cv2.imshow('jbjbjjb', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# print(fimalCountours)







# vid = cv2.VideoCapture(0)

# while (True):
#     ret, frame = vid.read()
#
#     cv2.imshow('frame', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# vid.release()
# cv2.destroyAllWindows()
