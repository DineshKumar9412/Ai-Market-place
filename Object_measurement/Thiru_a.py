# imports
import os, sys, time
import numpy as np
import cv2

# init camera
camera = cv2.VideoCapture(0)  ### <<<=== SET THE CORRECT CAMERA NUMBER
# camera.set(3, 1280)  # set frame width
# camera.set(4, 720)  # set frame height
# time.sleep(0.5)
# print(camera.get(3), camera.get(4))
# print(camera.get(5))

# master frame
master = None

while 1:

    # grab a frame
    (grabbed, frame0) = camera.read()

    # end of feed
    if not grabbed:
        break

    # gray frame
    frame1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    # blur frame
    frame2 = cv2.GaussianBlur(frame1, (15, 15), 0)

    # # initialize master
    # if master is None:
    #     master = frame2
    #     continue
    #
    # # delta frame
    # # frame3 = cv2.absdiff(master, frame2)

    # threshold frame
    frame4 = cv2.threshold(frame2, 15, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes
    kernel = np.ones((2, 2), np.uint8)
    frame5 = cv2.erode(frame4, kernel, iterations=4)
    frame5 = cv2.dilate(frame5, kernel, iterations=8)

    # find contours on thresholded image
    contours, nada = cv2.findContours(frame5.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # make coutour frame
    frame6 = frame0.copy()

    # target contours
    targets = []

    # loop over the contours
    for c in contours:

        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 500:
            continue

        # contour data
        M = cv2.moments(c)  # ;print( M )
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        x, y, w, h = cv2.boundingRect(c)
        rx = x + int(w / 2)
        ry = y + int(h / 2)
        ca = cv2.contourArea(c)

        # plot contours
        cv2.drawContours(frame6, [c], 0, (0, 0, 255), 2)
        # cv2.rectangle(frame6, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame6, (cx, cy), 2, (0, 0, 255), 10)
        # cv2.circle(frame6, (rx, ry), 2, (0, 255, 0), 10)

        # save target contours
        targets.append((cx, cy, ca))

    # key delay and action
    cv2.imshow('frame', frame6)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key != 255:
        print('key:', [chr(key)])

# release camera
camera.release()

# close all windows
cv2.destroyAllWindows()
