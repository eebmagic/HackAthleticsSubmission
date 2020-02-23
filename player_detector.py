import cv2
import numpy as np
from people_classifier.classify_team import classify_person
from util import get_frame

video = cv2.VideoCapture('media/1904-GATC-CONT-vs-PATE.mp4')

video.set(cv2.CAP_PROP_POS_FRAMES, get_frame('2:10'))

success, frame = video.read()
count = 0
success = True
idx = 0

field_sample = cv2.imread('media/field_sample.png') # load the base field image
field_hsv = cv2.cvtColor(field_sample, cv2.COLOR_BGR2HSV) # convert the field image into hsv
mu, sig = cv2.meanStdDev(field_hsv) # find the mean colour of the base field image
devs = 15 # tolerance

if __name__ == '__main__':
    while video.isOpened():
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # convert frame into hsv

        field_mask = cv2.inRange(hsv, mu - devs * sig, mu + devs * sig)
        masked = cv2.bitwise_and(frame, frame, mask=field_mask)

        # convert to hsv and to grayscale
        res_bgr = cv2.cvtColor(masked, cv2.COLOR_HSV2BGR)
        res_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

        # some magic to get a better output frame
        kernel = np.ones((13, 13), np.uint8)
        thresh = cv2.threshold(res_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contours in the threshold frame

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            # time to find the players
            if h >= 1.2 * w:
                if w > 50 and h >= 50:
                    dx = idx + 1
                    player_img = frame[y:(y + h), x:(x + w)]
                    # here we would call classify team
                    # classify_person(player_img)

                    cv2.rectangle(frame, (x, y),(x + w, y + h), (255, 0, 0), 3)


        cv2.imshow('output', frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        success, frame = video.read()

    video.release()
    cv2.destroyAllWindows()