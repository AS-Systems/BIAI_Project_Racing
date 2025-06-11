import cv2 
import numpy as np
from config import FRAME_STACK

#frame to grayscale, resize and normalization
def preprocess_frame(frame): 
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame,(84,84),interpolation=cv2.INTER_AREA)
    return frame/255.0

def stac_frame(frames, new_frame):
    if frames is None:
        return np.stack([new_frame]* FRAME_STACK, axis = 0)
    else:
        return np.append(frames[1:], [new_frame], axis = 0)
