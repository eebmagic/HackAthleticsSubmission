FRAME_RATE = 59.940024 # frame rate of the video

def get_frame(time):
    mini, sec = list(map(int, time.split(':')))
    return FRAME_RATE * (mini * 60 + sec)