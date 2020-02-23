import cv2
from imutils.object_detection import non_max_suppression
import numpy as np
from people_classifier.classify_team import classify_person
from util import get_frame, format_frame

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

video = cv2.VideoCapture('media/1904-GATC-CONT-vs-PATE.mp4')

video.set(cv2.CAP_PROP_POS_FRAMES, get_frame('2:10'))

success, frame = video.read()

field_sample = cv2.imread('media/field_sample.png') # load the base field image
field_hsv = cv2.cvtColor(field_sample, cv2.COLOR_BGR2HSV) # convert the field image into hsv
mu, sig = cv2.meanStdDev(field_hsv) # find the mean colour of the base field image
devs = 16 # tolerance

dark_field_sample = cv2.imread('media/dark_field_sample.png')
dark_field_hsv = cv2.cvtColor(dark_field_sample, cv2.COLOR_BGR2HSV)
dmu, dsig = cv2.meanStdDev(dark_field_hsv)
ddevs = 14

if __name__ == '__main__':
    while video.isOpened():
        orig = frame.copy()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # convert frame into hsv

        field_mask = cv2.inRange(hsv, mu - devs * sig, mu + devs * sig)
        dark_field_mask = cv2.inRange(hsv, dmu - ddevs * dsig, dmu + ddevs * dsig)
        masked = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(field_mask | dark_field_mask))

        # convert to hsv and to grayscale
        frame_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

        # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = hog.detectMultiScale(format_frame(frame_gray, 0.5), winStride=(8,8))

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        print(boxes)

        for (xA, yA, xB, yB) in boxes:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        cv2.imshow('output', frame)
        cv2.imshow('masked', masked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        success, frame = video.read()

    video.release()
    cv2.destroyAllWindows()