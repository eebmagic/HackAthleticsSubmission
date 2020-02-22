############################################################
### REFERENCES ####

# for pytessearct OCR examples
# https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/s

# pytesseract config string options
# https://ai-facets.org/tesseract-ocr-best-practices/

############################################################

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression  # for ignoring some boxes???
import pytesseract  # for OCR


# Define the detector to be used in the frames
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load the video 
cap = cv2.VideoCapture("/Users/ethanbolton/Desktop/VideoExample/videos/short_example.mp4")

counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    orig = frame.copy()

    if counter % 1 == 0:
        # Run detection on frame
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # Draw rectangles
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        for (xA, yA, xB, yB) in pick:
            print(xA, yA, xB, yB)

            roi = orig[yA:yB, xA:xB]

            # config = ("-l eng --oem 1 --psm 7")  # default config given from OCR example page
            # config = ("-l eng --oem 3 --psm 12 -c tessedit_char_whitelist='0123456789'")  # my optimal config based on article
            config = ("-l eng --oem 1 --psm 12 -c tessedit_char_whitelist='0123456789'")  # my optimal config based on article
            # text = pytesseract.image_to_string(roi, config=config)
            
            # if text != "":
            #     cv2.imshow(text, roi)
            #     print(f"Detected Text: {text}")

                # input("press enter for next frame:\n")


            # Draw rectangles on frame ?
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # input("press enter for next frame:\n")
            
    counter += 1

    # Show the finished frame
    # cv2.imshow('original', orig)
    cv2.imshow("with rects", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
