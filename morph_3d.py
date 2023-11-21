from random import sample
import numpy as np
from wand.image import Image
import cv2
import matplotlib.pyplot as plt
import bezier
import os 

kernel_3 = np.ones((3, 3), np.uint8)
kernel_5 = np.ones((5, 5), np.uint8)
kernel_7 = np.ones((7, 7), np.uint8)
kernel_11 = np.ones((11, 11), np.uint8)

moving_direction = np.array([[-1, -1], [ -1,0], [-1, 1], [0, 1],
                              [1, 1], [1, 0], [1, -1], [0, -1]])
wdxa = np.array([0,1,0,1,0,1,0,1])
qecz = np.array([1,0,1,0,1,0,1,0])

lineEnds = """
    3>:
        0,0,-
        0,1,1
        0,0,-;
    3>:
        0,0,0
        0,1,0
        0,0,1
    """
    
lineJunctions = """
    3>:
        1,-,1
        -,1,-
        -,1,-;
    3>:
        -,1,-
        -,1,1
        1,-,-;
    3>:
        1,-,-
        -,1,-
        1,-,1
    """

def read_dlo(dlo_image_name, manual_mask = None, vis = False, occ = False):
    dlo = cv2.imread(dlo_image_name)
    dlo = dlo[:, :, 0]
    dlo_pad = np.zeros((1080,1080))
    if dlo.shape[0] == 1080 and dlo.shape[1] == 1920:
        dlo_pad[12:-12, 12:-12] = dlo[12:-12, 432:-432]
    else:
        dlo_pad[12:-12, 12:-12] = dlo
    dlo_pad = dlo_pad.astype(np.uint8)

    dlo_dilated = cv2.dilate(dlo_pad, kernel_5)
    #dlo_dilated = cv2.erode(dlo_pad, kernel_3)
    #dlo_dilated = dlo_pad

    if manual_mask is not None:
        dlo_dilated = dlo_dilated * manual_mask
    
    if vis == True:
        plt.imshow(dlo_pad)
        plt.show()
    return dlo_dilated, dlo_pad

def kernel_reorder(kernel, last_index):
    reordered_kernel = np.zeros((8,))
    k = 0
    # 8 directions
    for i,j in [[0,0], [0,1], [0,2], [1,2], [2,2], [2,1], [2,0], [1,0]]:
        reordered_kernel[k]=kernel[i,j]  
        k += 1
    if last_index == -1:
        return reordered_kernel
    current_index = (last_index - 4) % 8
    reordered_kernel[current_index] = 0
    return reordered_kernel


def get_skeleton(dlo, erode = True, vis = False):
    if erode == True:
        dlo = cv2.erode(dlo, kernel_5)
    
    dlo_thinned = cv2.ximgproc.thinning(dlo)
    
    if vis == True:
        plt.imshow(dlo_thinned)
        plt.show()
    return dlo_thinned