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


def contrast(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()

    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf[img]



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

        img3 = cv.resize(img, (track_window[2], track_window[3]))

        img2[track_window[1]:track_window[1] + track_window[3], track_window[0]:track_window[0] + track_window[2]] = img3

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