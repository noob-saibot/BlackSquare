import numpy as np
import cv2 as cv
from PIL import Image
cap = cv.VideoCapture('videos/example_video.avi')
# cap = cv.VideoCapture('videos/slow_traffic_small.mp4')
# take first frame of the video
ret,frame = cap.read()
# setup initial location of window
# r,h,c,w = 250,90,400,125  # simply hardcoded the values
x, y, w, h = np.round(cv.selectROI(frame, False)).astype(int)
track_window = (x, y, w, h)
# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
img = cv.imread('videos/pepsi.jpg')
im = Image.open('videos/pepsi.jpg')
while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv.CamShift(dst, track_window, term_crit)

        print(track_window)

        rows, cols, _ = img.shape

        # Draw it on image
        timer = cv.getTickCount()
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
        cv.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame, [pts], True, 255, 2)
        print(pts)

        if pts[0][0] > pts[3][0]:
            x2 = pts[0][0]
            x1 = pts[3][0]
        else:
            x1 = pts[0][0]
            x2 = pts[3][0]

        if pts[2][1] > pts[0][1]:
            y1 = pts[0][1]
            y2 = pts[2][1]
        else:
            y2 = pts[0][1]
            y1 = pts[2][1]

        print(x2 - x1, y2 - y1)

        img3 = cv.resize(img, (x2 - x1, y2 - y1))

        print(x1, x2, y1, y2)

        img2[y1:y2, x1:x2] = img3

        cv.imshow('img3', img3)
        cv.imshow('img2', img2)

        k = cv.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv.imwrite(chr(k)+".jpg",img2)
    else:
        break
cv.destroyAllWindows()
cap.release()