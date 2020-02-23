import cv2
import numpy as np

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


if __name__ == '__main__':
    import os
    prefix = "../scene_detection/outputs/"
    images = os.listdir(prefix)
    images.sort()
    paths = [prefix + x for x in images]

    for IMAGE_PATH in paths[:3]:
        image = cv2.imread(IMAGE_PATH)
        print(IMAGE_PATH.split('/')[-1], common_colors(image))



    # used for testing but left as an example

    # SOURCE_PATH = "/Users/ethanbolton/Desktop/hackgt/scene_detection/outputs/22360.png"
    # image = cv2.imread(SOURCE_PATH)


    # print(image[0][0])
    # print(image.shape)

    # colors = get_crosses(image)
    # print(colors)

    # flatList = []
    # for color in colors:
    #     as_list = color.tolist()
    #     flatList.append(as_list)

    # print(avg_pixel(flatList))

    # print(type(image))
    # print(type(flatList))
