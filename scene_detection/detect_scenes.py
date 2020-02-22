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


def get_change_number(frameA, frameB):
    # pulled from my previous motion detection project
    # probably not optimal for this case?
    DIFF = cv2.absdiff(np.float32(frameA), np.float32(frameB))
    MAX_DIFF = np.amax(DIFF)
    # cv2.imshow("absdiff", DIFF)

    return MAX_DIFF


# SAVING_IMAGES = True

VIDEO_PATH = "/Users/ethanbolton/Desktop/hackgt/video/GT_video/short.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)


last_frame = format_frame(cap.read()[1])
counter = 0
previous_change = None
last_significant_framedata = np.empty(last_frame.shape)
print("LAST SIGNIFICANT")
print(last_significant_framedata.any())
running_list = []

saveCounter = None

while cap.isOpened():
    if counter % 1000 == 0:
        print(running_list)

    frame_iterations = 20
    if counter % frame_iterations == 0:
        ret, current_frame = cap.read()
        current_frame = format_frame(current_frame)

        MAX_DIFF = get_change_number(last_frame, current_frame)

        # print(last_frame, current_frame)
        print(counter, MAX_DIFF)
        if previous_change != None and abs(MAX_DIFF - previous_change) > 60:
            
            # Compare last changed frame
            if not last_significant_framedata.any():
                AGAINST_LAST_SIG_FRAME = get_change_number(last_significant_framedata, current_frame)

            if not last_significant_framedata.any() or AGAINST_LAST_SIG_FRAME > 80:

                print(f"CHANGE_DETECTED: frame {counter}")

                
                if len(running_list) != 0 and counter - running_list[-1] > 1500:
                    running_list.append(counter)
                    last_significant_framedata = current_frame
                    print("LAST SIGNIFICANT")
                    print(last_significant_framedata.any())
                    saveCounter = counter + (3 * frame_iterations) # set svae to be in three more progressions
                elif len(running_list) == 0:
                    running_list.append(counter)
                    last_significant_framedata = current_frame

        if SAVING_IMAGES:
            if saveCounter != None and abs(saveCounter - counter) < frame_iterations:
                cv2.imwrite(f"outputs/{counter}.png", current_frame)


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
