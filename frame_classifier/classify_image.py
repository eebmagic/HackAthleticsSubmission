import cv2
import numpy as np

from most_common_color import avg_pixel
from most_common_color import common_colors

def is_board(image):
    # method using avg pixel color
    avg_color = avg_pixel(image)
    G_avg = avg_color[1]

    # method using three most common colors
    colors = common_colors(image)
    G_sum = sum([x[0][1] for x in colors])

    # print(avg_color)
    # print(G_avg)
    # print(colors)
    # print(G_sum)

    ## Compromise between two methods
    # if G_avg < 150 and G_sum < 150:
    # if G_avg < 150 and G_sum < 150 and colors[0][1] < 265:
    if G_avg < 150 and colors[0][1] < 265:
        return True
    else:
        return False


if __name__ == '__main__':
    import os
    prefix = "../scene_detection/outputs/"
    images = os.listdir(prefix)
    images.sort()
    paths = [prefix + x for x in images]

    correct = ["43760.png", "104280.png", "131240.png"]

    for IMAGE_PATH in paths:
        image = cv2.imread(IMAGE_PATH)
        
        prediction = is_board(image)
        print(IMAGE_PATH.split('/')[-1], prediction)
        if IMAGE_PATH.split('/')[-1] in correct:
            print(f"BOARD - {prediction == True}")
        else:
            print(f"FIELD - {prediction == False}")
        print("")