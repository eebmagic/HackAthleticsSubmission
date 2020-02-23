import cv2
import numpy as np
import os

# line_img = cv2.imread('rough_prototype/line_image.png') # load the base line image
# line_hsv = cv2.cvtColor(line_img, cv2.COLOR_BGR2HSV) # convert the line image into hsv
# mu, sig = cv2.meanStdDev(line_hsv) # find the mean colour of the base line image
# devs = 15 # tolerance

def get_angle(image, crop_factor=0.1):
    (h, w) = image.shape[:2]
    image = image[int(h * crop_factor):int(h * (1 - crop_factor)), int(w * crop_factor):int(w * (1 - crop_factor))] # crop the image to remove the sidelines
    
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # line_mask = cv2.inRange(hsv, mu - devs * sig, mu + devs * sig)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 85)

    if len(lines) <= 0:
        return

    lines = np.reshape(lines, (len(lines), 2))

    average_angle = sum([x[1] for x in lines]) / len(lines)

    return average_angle

def is_wide(image, crop_factor=0.1):
    average_angle = get_angle(image, crop_factor)

    if average_angle < (3 * np.pi / 4) and average_angle > (np.pi / 4):
        return True
    else:
        return False

    # print(lines)

    # this will draw all the lines
    # for rho, theta in lines:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * a)
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * a)

    #     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # cv2.imshow('generated_lines', image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# expected output
# These should be vertical and the rest should be horizontal
# 77980 
# 80460
# 106720

if __name__ == '__main__':
    prefix = 'scene_detection/outputs/'
    images = os.listdir(prefix)
    images.sort()

    for image in images:
        print(f'{image} : {is_wide(cv2.imread(prefix + image))}')
