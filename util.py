FRAME_RATE = 59.940024 # frame rate of the video

def get_frame(time):
    mini, sec = list(map(int, time.split(':')))
    return FRAME_RATE * (mini * 60 + sec)


def get_time(frameCounter):
    secs = frameCounter / FRAME_RATE
    mins = int(secs // 60)
    loose_secs = int(secs - (mins * 60))

    return f"{mins}:{loose_secs}"