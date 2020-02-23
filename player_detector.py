import cv2
import cvlib as cv
import numpy as np
from people_classifier.classify_team import classify_person
from cvlib.object_detection import draw_bbox
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

line_sample = cv2.imread('media/line_sample.png')
line_hsv = cv2.cvtColor(line_sample, cv2.COLOR_BGR2HSV)
lmu, lsig = cv2.meanStdDev(line_hsv)
ldevs = 10

if __name__ == '__main__':
    while video.isOpened():
        frame_out = frame.copy()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # convert frame into hsv

        field_mask = cv2.inRange(hsv, mu - devs * sig, mu + devs * sig)
        dark_field_mask = cv2.inRange(hsv, dmu - ddevs * dsig, dmu + ddevs * dsig)
        line_mask = cv2.inRange(hsv, lmu - ldevs * lsig, lmu + ldevs * lsig)
        masked = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(field_mask | dark_field_mask | line_mask))

        # convert to hsv and to grayscale
        frame_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

        frame = format_frame(frame, 0.8)

        bboxs, labels, conf = cv.detect_common_objects(frame)
        # frame = draw_bbox(frame, bboxs, labels, conf)

        combined = [(bboxs[x], labels[x]) for x in range(0, len(labels))]

        for bbox, label in combined:
            if label == 'person':
                # cv2.imwrite(f'media/players/{bbox[0]}.png', frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])
                player = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if classify_person(player):
                    print('TECH!!')
                    
                cv2.rectangle(frame_out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        cv2.imshow('output', frame_out)
        cv2.imshow('masked', masked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        success, frame = video.read()

    video.release()
    cv2.destroyAllWindows()