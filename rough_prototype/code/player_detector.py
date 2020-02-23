import cv2
import numpy as np
from util import get_frame

video = cv2.VideoCapture('media/1904-GATC-CONT-vs-PATE.mp4')

video.set(cv2.CAP_PROP_POS_FRAMES, get_frame('2:10'))

success, frame = video.read()
count = 0
success = True
idx = 0

field_sample = cv2.imread('media/field_sample.png') # load the base field image
field_hsv = cv2.cvtColor(field_sample, cv2.COLOR_BGR2HSV) # convert the field image into hsv
mu, sig = cv2.meanStdDev(field_hsv) # find the mean colour of the base field image
devs = 15 # tolerance

if __name__ == '__main__':
    while success:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # convert frame into hsv

        field_mask = cv2.inRange(hsv, mu - devs * sig, mu + devs * sig)
        masked = cv2.bitwise_and(frame, frame, mask=field_mask)

        # convert to hsv and to grayscale
        res_bgr = cv2.cvtColor(masked, cv2.COLOR_HSV2BGR)
        res_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

        # some magic to get a better output image
        kernel = np.ones((13, 13), np.uint8)
        thresh = cv2.threshold(res_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        cv2.imshow('output', frame)
        cv2.waitKey(0)