import cv2
import numpy as np
import time

from util import get_time

from frame_classifier import classify_board
from frame_classifier import orientation_detector


'''
correct answers:

0    - 3 secs  - board
160  - 19 secs - wide
1115 - 35 secs - long
2074 - 38 secs - board 

'''

# Used for getting specifc frame configuration to optimize time spent for finding scenes
def format_frame(inputFrame, percent=20):
    # reduce resolution
    width = int(inputFrame.shape[1] * percent/ 100)
    height = int(inputFrame.shape[0] * percent/ 100)
    dim = (width, height)

    return cv2.resize(inputFrame, dim, interpolation=cv2.INTER_AREA)


def mode_from_frame(inputFrame):
    output = None
    if classify_board.is_board(inputFrame, printouts=True) == True:
        print("VAR UPDATED TO - BOARD")
        output = "board"
    else:
        print("NOT BOARD")
        is_wide = orientation_detector.is_wide(inputFrame, crop_factor=0.3, printouts=False)
        if is_wide == "wide":
            print("VAR UPDATED TO - WIDE")
            output = "wide"
        elif is_wide == "long":
            print("VAR UPDATED TO - LONG")
            output = "long"
        else:
            output = is_wide

    # cv2.imshow(str(inputFrame), inputFrame)
    # input()
    return output


def get_frame_stamps(imagePath, SAVE_TRANSITION_FRAMES=False):
    cap = cv2.VideoCapture(VIDEO_PATH)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 120)

    frame_counter = 0

    last_frame = format_frame(cap.read()[1])
    previous_change = None

    last_significant_framedata = np.empty(last_frame.shape)
    running_list = []

    # mean data for detecting scene changes
    # (value, count since last transition)
    mean_data = [last_frame.mean(), 1]
    MEAN_THRESH = 2

    next_counter = None

    # modes: board, wide, long
    current_shot_mode = None

    # how many frames to do before checking all things
    frame_iterations = 1

    while cap.isOpened():
        ret, current_full_frame = cap.read()
        if ret:

            if frame_counter % 1000 == 0:
                print(running_list)

            #################################################
            ### Check for updated scenes 
            if frame_counter % frame_iterations == 0:    
                current_frame = format_frame(current_full_frame)

                # display every few frames for ease
                if frame_counter % 1 == 0:
                    cv2.imshow(current_shot_mode, current_frame)
                    pass


                # Check mean for motion
                curr_mean = current_frame.mean()
                print(current_shot_mode, get_time(frame_counter), frame_counter,  round(mean_data[0]), round(curr_mean))
                if abs(curr_mean - mean_data[0]) > MEAN_THRESH:
                    # update new scene
                    print("CHANGE_DETECTED")
                    running_list.append(frame_counter)
                    next_counter = frame_counter + (10 * frame_iterations) # next frame to check for save and update state
                    mean_data = [curr_mean, 1]
                else:
                    # update avg
                    mean_data[0] = ((mean_data[0] * mean_data[1]) + curr_mean) / (mean_data[1] + 1)
                    mean_data[1] += 1

                
                # update current mode and/or save frames
                if next_counter != None and abs(next_counter - frame_counter) < frame_iterations:
                    # update shot mode after delay
                    current_shot_mode = mode_from_frame(current_frame)

                    if SAVE_TRANSITION_FRAMES:
                        cv2.imwrite(f"outputs/{frame_counter}.png", current_frame)


                #################################################
                ### send frame info to proper processors
                if current_shot_mode == "board":
                    pass
                
                elif current_shot_mode == "wide":
                    pass
                
                elif current_shot_mode == "long":
                    pass
                
                # Send to frame classifier
                else:
                    current_shot_mode = mode_from_frame(current_frame)



            #################################################
            ### Update before moving to next frame 
            last_frame = current_frame
            # previous_change = MAX_DIFF
            frame_counter += 1
            if cv2.waitKey(1) == ord('q'):
                break
            #################################################
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    VIDEO_PATH = "media/short.mp4"
    # VIDEO_PATH = "/Users/ethanbolton/Desktop/super_short.mp4"
    get_frame_stamps(VIDEO_PATH)