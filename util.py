import cv2

FRAME_RATE = 59.940024 # frame rate of the video

def get_frame(time):
    mini, sec = list(map(int, time.split(':')))
    return FRAME_RATE * (mini * 60 + sec)

def format_frame(image, percent=20):
    # reduce color to grayscale
    # inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2GRAY)

    # reduce resolution
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    dim = (width, height)

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)