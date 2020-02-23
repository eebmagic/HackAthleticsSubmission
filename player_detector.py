import cv2
import numpy as np
from frame_classifier.orientation_detector import get_lines
from people_classifier.classify_team import classify_person
from util import get_frame, format_frame

video = cv2.VideoCapture('media/1904-GATC-CONT-vs-PATE.mp4')

video.set(cv2.CAP_PROP_POS_FRAMES, get_frame('2:10'))

success, frame = video.read()
(h, w) = frame.shape[:2]

count = 0
success = True
idx = 0

field_sample = cv2.imread('media/field_sample.png') # load the base field image
field_hsv = cv2.cvtColor(field_sample, cv2.COLOR_BGR2HSV) # convert the field image into hsv
mu, sig = cv2.meanStdDev(field_hsv) # find the mean colour of the base field image
devs = 14 # tolerance

dark_field_sample = cv2.imread('media/dark_field_sample.png')
dark_field_hsv = cv2.cvtColor(dark_field_sample, cv2.COLOR_BGR2HSV)
dmu, dsig = cv2.meanStdDev(dark_field_hsv)
ddevs = 14

def generate_line_mask(lines):
    mask = np.zeros((h, w), np.uint8)
    mask.fill(255)

    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho * 10
        y0 = b * rho * 10
        x1 = int(x0 + 1000 * (-b) * 10)
        y1 = int(y0 + 1000 * a * 10)
        x2 = int(x0 - 1000 * (-b) * 10)
        y2 = int(y0 - 1000 * a * 10)

        cv2.line(mask, (x1, y1), (x2, y2), 0, 3)

    cv2.imshow('lines', mask)
    return mask

if __name__ == '__main__':
    while video.isOpened():
        (h, w) = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # convert frame into hsv

        line_mask = generate_line_mask(get_lines(format_frame(frame, 10)))
        field_mask = cv2.inRange(hsv, mu - devs * sig, mu + devs * sig)
        dark_field_mask = cv2.inRange(hsv, dmu - ddevs * dsig, dmu + ddevs * dsig)
        masked = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(field_mask | dark_field_mask | line_mask))

        # convert to hsv and to grayscale
        frame_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

        # some magic to get a better output frame
        kernel = np.ones((13, 13), np.uint8)
        _, thresh = cv2.threshold(frame_gray, 25, 255, 0)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contours in the threshold frame
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            # time to find the players
            if h >= 1.2 * w and h <= 5 * w:
                if w > 50 and h >= 50:
                    dx = idx + 1
                    player_img = frame[y:(y + h), x:(x + w)]
                    # here we would call classify team
                    # classify_person(player_img)

                    cv2.rectangle(frame, (x, y),(x + w, y + h), (255, 0, 0), 3)


        cv2.imshow('output', frame)
        cv2.imshow('masked', masked)
        cv2.imshow('gray', frame_gray)
        cv2.imshow('thresh', thresh)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        success, frame = video.read()

    video.release()
    cv2.destroyAllWindows()