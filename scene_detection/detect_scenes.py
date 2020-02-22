import cv2
import numpy as np


'''
correct answers:

3,200  - 3 secs
22,300 - 19 secs
41,480 - 35 secs
44,800 - 38 secs

'''

# Used for getting specifc frame configuration to optimize time spent for finding scenes
def format_frame(inputFrame, percent=20):
    # reduce color to grayscale
    # inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2GRAY)

    # reduce resolution
    width = int(inputFrame.shape[1] * percent/ 100)
    height = int(inputFrame.shape[0] * percent/ 100)
    dim = (width, height)

    return cv2.resize(inputFrame, dim, interpolation=cv2.INTER_AREA)


VIDEO_PATH = "/Users/ethanbolton/Desktop/hackgt/video/GT_video/short.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

last_frame = format_frame(cap.read()[1])
counter = 0
previous_change = None
running_list = []

while cap.isOpened():
    if counter % 1000 == 0:
        print(running_list)

    if counter % 20 == 0:
        ret, current_frame = cap.read()
        current_frame = format_frame(current_frame)

        # pulled from my previous motion detection project
        # probably not optimal for this case?
        DIFF = cv2.absdiff(np.float32(last_frame), np.float32(current_frame))
        # cv2.imshow("absdiff", DIFF)
        MAX_DIFF = np.amax(DIFF)

        # print(last_frame, current_frame)
        print(counter, MAX_DIFF)
        if previous_change != None and abs(MAX_DIFF - previous_change) > 60:
            print(f"CHANGE_DETECTED: frame {counter}")

            
            if len(running_list) != 0 and counter - running_list[-1] > 1500:
                running_list.append(counter)
            elif len(running_list) == 0:
                running_list.append(counter)



        # makes significantly slower
        # cv2.imshow("resized", last_frame)
        # input("press enter for next frame")

    # Cycle before next frame
    last_frame = current_frame
    previous_change = MAX_DIFF
    counter += 1
    if not KeyboardInterrupt:
        print(running_list)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
