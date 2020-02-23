import cv2
import numpy as np
import os
import math


def get_angle(image, crop_factor=0.2, printlines=False, showlines=False):
    (h, w) = image.shape[:2]
    # crop the image to remove the sidelines
    cropped = image[int(h * crop_factor):int(h * (1 - crop_factor)), int(w * crop_factor):int(w * (1 - crop_factor))] 
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV) # convert image into hsv

    # make mask for lines
    line_sample = cv2.imread('./media/line_sample.png')
    line_hsv = cv2.cvtColor(line_sample, cv2.COLOR_BGR2HSV)
    lmu, lsig = cv2.meanStdDev(line_hsv)
    ldevs = 18

    line_mask = cv2.inRange(hsv, lmu - ldevs * lsig, lmu + ldevs * lsig)

    # use erosion on line_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    thresh = cv2.erode(line_mask, kernel)
    thresh = cv2.dilate(thresh, kernel)
    thresh = cv2.dilate(thresh, kernel)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) # converts the image to grayscale
    masked = cv2.bitwise_and(gray, gray, mask=line_mask)
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3) # find the edges in the image

    thresh_mask = cv2.bitwise_and(masked, masked, mask=cv2.bitwise_not(thresh))

    # for debugging
    # cv2.imshow("cropped", image)
    # cv2.imshow("gray", gray)
    # cv2.imshow("line_mask", line_mask)
    # cv2.imshow("thresh", thresh)
    # cv2.imshow("thresh_mask", thresh_mask)
    # cv2.imshow("masked", masked)
    # cv2.imshow("edges", edges)
    lines = cv2.HoughLines(thresh_mask, 1, np.pi / 180, 35) # find the lines

    if lines is None:
        print("NO LINES FOUND")
        return

    elif len(lines) <= 0:
        print("NO LINES FOUND")
        return

    lines = np.reshape(lines, (len(lines), 2))
    average_angle = sum([x[1] for x in lines]) / len(lines)
    average_degree = math.degrees(average_angle)

    if printlines:
        print(lines)
        print(f"number of lines: {len(lines)}")
        print(f"average_angle: {average_angle}")
        print(f"average_degree: {average_degree}")

    if showlines:
        # this will draw all the lines
        print("MAKING LINES IMAGE")
        line_layover = cropped.copy()
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)

            cv2.line(line_layover, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow('generated_lines', line_layover)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return average_angle


def is_wide(image, crop_factor=0.2, printouts=False):
    average_angle = get_angle(image, crop_factor, printlines=printouts, showlines=False)
    # average_degree = math.degrees(average_angle)
    if printouts:
        print(f"average_angle: {average_angle}")
        # print(f"average_degree: {average_degree}")

    


    if average_angle is None:
        return "orientation_detector ERROR: no lines found"

    if average_angle < (3 * np.pi / 4) and average_angle > (np.pi / 4): # check if the average angle is horizontal
        return "long"
    else:
        return "wide"

    

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
