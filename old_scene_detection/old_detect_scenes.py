import cv2
import numpy as np

import sys
sys.path.append("/Users/ethanbolton/Desktop/hackgt/frame_classifier")
# from frame_classifier import classify_is_board

import frame_classifier.classify_is_board

quit("this file is an old version. Use the newly placed one that makes imports easier")

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

def get_frame_stamps(imagePath, SAVE_TRANSITION_FRAMES=False):
    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_counter = 0

    last_frame = format_frame(cap.read()[1])
    previous_change = None

    last_significant_framedata = np.empty(last_frame.shape)
    running_list = []

    save_counter = None

    # modes: board, wide, long
    current_shot_mode = None

    while cap.isOpened():
        if frame_counter % 1000 == 0:
            print(running_list)

        frame_iterations = 20
        if frame_counter % frame_iterations == 0:
            ret, current_frame = cap.read()
            current_frame = format_frame(current_frame)

            MAX_DIFF = get_change_number(last_frame, current_frame)

            # print(last_frame, current_frame)
            print(frame_counter, MAX_DIFF)
            if previous_change != None and abs(MAX_DIFF - previous_change) > 60:
                
                # Compare last changed frame
                if not last_significant_framedata.any():
                    AGAINST_LAST_SIG_FRAME = get_change_number(last_significant_framedata, current_frame)

                if not last_significant_framedata.any() or AGAINST_LAST_SIG_FRAME > 80:

                    print(f"CHANGE_DETECTED: frame {frame_counter}")

                    
                    if len(running_list) != 0 and frame_counter - running_list[-1] > 1500:
                        running_list.append(frame_counter)
                        last_significant_framedata = current_frame
                        print("LAST SIGNIFICANT")
                        print(last_significant_framedata.any())
                        save_counter = frame_counter + (3 * frame_iterations) # set svae to be in three more progressions
                    elif len(running_list) == 0:
                        running_list.append(frame_counter)
                        last_significant_framedata = current_frame

            if SAVE_TRANSITION_FRAMES:
                if save_counter != None and abs(save_counter - frame_counter) < frame_iterations:
                    cv2.imwrite(f"outputs/{frame_counter}.png", current_frame)




        # Cycle values before next frame
        last_frame = current_frame
        previous_change = MAX_DIFF
        frame_counter += 1
        if not KeyboardInterrupt:
            print(running_list)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    VIDEO_PATH = "../video/GT_video/short.mp4"
    get_frame_stamps(VIDEO_PATH)