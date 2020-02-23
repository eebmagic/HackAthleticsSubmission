import cv2
import numpy as np
import os

temple_sample = cv2.imread('media/temple_sample.png') # load the base jersey image
temple_hsv = cv2.cvtColor(temple_sample, cv2.COLOR_BGR2HSV) # convert the jersey image into hsv
mu, sig = cv2.meanStdDev(temple_hsv) # find the mean colour of the base jersey image
devs = 1 # tolerance

def classify_person(image, red_threshold=250):
    if image is None:
        return False
    if image.size == 0:
        return False
        
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, mu - devs * sig, mu + devs * sig)
    masked = cv2.bitwise_and(image, image, mask=red_mask)

    if cv2.countNonZero(red_mask) > red_threshold:
        return False
    else:
        return True

    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    prefix = 'media/players/'
    images = os.listdir(prefix)
    images.sort()

    for image in images:
        print(f'{image} : {classify_person(cv2.imread(prefix + image))}')
