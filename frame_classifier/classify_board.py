'''
Classifier to determine if an image is of scoreboard or of field
'''

import cv2
import numpy as np
import os

def is_board(image, printouts=False):
    # method using avg pixel color
    avg_color = avg_pixel(image)
    G_avg = avg_color[1]

    # method using three most common colors
    colors = common_colors(image)
    G_sum = sum([x[0][1] for x in colors])
    max_count = colors[0][1]
    R_max = colors[0][0][0]
    G_max = colors[0][0][1]
    B_max = colors[0][0][2]
    RB_max_avg = int((R_max + B_max) / 2)

    if printouts:
        # print("")
        print(avg_color)
        print(f"G_avg: {G_avg}")
        print(colors)
        print(f"max_count: {max_count}")
        print(f"G_sum: {G_sum}")
        print(f"G_max: {G_max}")
        print(f"RB_max_avg: {RB_max_avg}")

    ## Compromise between two methods
    # if G_avg < 150 and G_sum < 150:
    # if G_avg < 150 and G_sum < 150 and colors[0][1] < 265:
    if G_avg < 150 and max_count < 300 and (G_max - RB_max_avg) < 15:
        return True
    else:
        return False


def get_crosses(inputImage):
    output = []
    height, width, depth = inputImage.shape
    ratio = width / height
    y = 0

    for x in range(width):
        # make pixels white for example of path
        # inputImage[y][x] = np.array([254, 254, 254])
        # inputImage[(height - 1) - y][x] = np.array([254, 254, 254])
        # inputImage[int((height - 1) / 2)][x] = np.array([254, 254, 254])
        # inputImage[(height - 1) - y][int((width - 1) / 2)] = np.array([254, 254, 254])

        output.append(inputImage[y][x])
        output.append(inputImage[(height - 1) - y][x])
        output.append(inputImage[int((height - 1) / 2)][x])
        output.append(inputImage[(height - 1) - y][int((width - 1) / 2)])

        y = int(x * (height / width))

    return output

# takes a python list of color data and returns the average color
def avg_pixel(inputList):
    if type(inputList) != list:
        inputList = get_crosses(inputList)

    R, G, B = 0, 0, 0
    
    for color in inputList:
        R += color[0]
        G += color[1]
        B += color[2]
    
    R = int(R / len(inputList))
    G = int(G / len(inputList))
    B = int(B / len(inputList))

    return (R, G, B)


def round_colorlist(inputList, factor=20):
    newData = []
    for point in inputList:
        newTup = [round(x - (x % factor), -1) for x in point[:3]]
        newData.append(tuple(newTup))

    return newData


# gets most common rather than average
def common_colors(inputList):
    if type(inputList) != list:
        inputList = get_crosses(inputList)

    rounded = round_colorlist(inputList)
    counts = {}
    for color in rounded:
        if color not in counts:
            counts[color] = 1
        else:
            counts[color] += 1
    counts = sorted(counts.items(), key=lambda x: int(x[1]), reverse=True)

    short = counts[:3]
    shortValues = [x[0] for x in short]
    return short


# if __name__ == '__main__':
#     import os
#     prefix = "../old_scene_detection/outputs/"
#     images = os.listdir(prefix)
#     images.sort()
#     paths = [prefix + x for x in images]

#     correct = ["43760.png", "104280.png", "131240.png"]

#     for IMAGE_PATH in paths:
#         image = cv2.imread(IMAGE_PATH)
        
#         prediction = is_board(image)
#         print(IMAGE_PATH.split('/')[-1], prediction)
#         if IMAGE_PATH.split('/')[-1] in correct:
#             print(f"BOARD - {prediction == True}")
#         else:
#             print(f"FIELD - {prediction == False}")
#         print("")