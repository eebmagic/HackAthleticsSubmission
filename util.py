<<<<<<< HEAD
import cv2

=======
>>>>>>> 9c0cfb54523b9983711203f38fdafc1497b7caf6
FRAME_RATE = 59.940024 # frame rate of the video

def get_frame(time):
    mini, sec = list(map(int, time.split(':')))
    return FRAME_RATE * (mini * 60 + sec)

<<<<<<< HEAD
def format_frame(image, factor=0.2):
    # reduce color to grayscale
    # inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2GRAY)

    # reduce resolution
    width = int(image.shape[1] * factor)
    height = int(image.shape[0] * factor)
    dim = (width, height)

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
=======

def get_time(frameCounter):
    secs = frameCounter / FRAME_RATE
    mins = int(secs // 60)
    loose_secs = int(secs - (mins * 60))

    return f"{mins}:{loose_secs}"
>>>>>>> 9c0cfb54523b9983711203f38fdafc1497b7caf6
