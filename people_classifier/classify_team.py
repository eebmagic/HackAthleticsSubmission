import cv2
import numpy as np
from skimage import io

def classify_person(image, observations=5):
    average = image.mean(axis=0).mean(axis=0)

    pixels = np.float32(image.reshape(-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # this takes n samples over k clusters to determine the averagest colours
    _, labels, palette = cv2.kmeans(pixels, observations, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    # mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
	# res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
	# res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
	# res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)