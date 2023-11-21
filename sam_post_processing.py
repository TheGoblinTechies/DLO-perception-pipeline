import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import measure
import cv2 
from wand.image import Image

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

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



IMG_PATH = 'xxx'
mask = np.load(IMG_PATH)
# plt.imshow(mask)
# plt.show()
# set all boundary pixels to 0
boundary_list = [0,1,2,3,4,5,6,7,8,9,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]
# center cropping
mask = mask[:, range(420, 1500)]
mask[boundary_list,:] = 0
mask[:,boundary_list] = 0
mask = np.where(mask > 0, 1, 0).astype(np.uint8)
mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=1)
# plt.imshow(mask)
# plt.show()

# extract all connected components of the mask

all_labels = measure.label(mask)

# get number of connected components
num_of_comp = np.max(all_labels)

print(all_labels.shape)
# plt.imshow(all_labels)
# plt.show()

comp_mask = np.where(all_labels > 0, 1, 0)
# eliminate all compoenents that are too small
# (i.e. smaller than 1% of the image area)
for j in range(1, num_of_comp + 1):
    if np.sum(all_labels == j) < 500:#0.001 * np.prod(mask.shape):
        comp_mask[all_labels == j] = 0


        
# plt.imshow(comp_mask)
# plt.show()
comp_mask *= 255
# get skeleton of each component by using opencv
comp_skeleton = cv2.ximgproc.thinning(comp_mask.astype(np.uint8))

#plt.imshow(comp_skeleton)
#plt.show()

skeleton_labels = measure.label(comp_skeleton)
skeleton_labels[[0,-1],:] = 0
skeleton_labels[:,[0,-1]] = 0
num_of_skeletons = np.max(skeleton_labels)

rope_like_skeletons = []
rope_like_length = []
for k in range(1, num_of_skeletons + 1):
    # get each compoent of comp_skeleton
    comp_skeleton_k = np.where(skeleton_labels == k, 255, 0).astype(np.uint8)

    rope_like_mask = np.zeros_like(comp_mask)
    cord_x = np.where(skeleton_labels == k)[0][0]
    cord_y = np.where(skeleton_labels == k)[1][0]
    #print(np.array(np.where(skeleton_labels == k)).shape, cordinates.shape)
    # find its corresponding pixel in all_labels
    for m in range(num_of_comp):
        label_m = all_labels[cord_x, cord_y]
        #print(k, m, cordinates, label_m)
        if label_m != 0:
            rope_like_mask[all_labels == label_m] = 1
            break
    # area of rope_like_mask
    area = np.sum(rope_like_mask > 0)
    length = np.sum(comp_skeleton_k > 0)
    width = area / length
    

    # get the ends of the skeleton
    endsImage_k = Image.from_array(comp_skeleton_k)
    endsImage_k.morphology(method='hit_and_miss', kernel=lineEnds)
    endsImage_k = np.array(endsImage_k)
    num_of_ends = np.sum(endsImage_k[:,:,0] > 0)
    
    # get the junctions of the skeleton
    junctionsImage_k = Image.from_array(comp_skeleton_k)
    junctionsImage_k.morphology(method='hit_and_miss', kernel=lineJunctions)
    junctionsImage_k = np.array(junctionsImage_k)
    num_of_junctions = np.sum(junctionsImage_k[:,:,0] > 0)
    

    print(k, num_of_ends, num_of_junctions)
    print(width,length, area)
    #visualize(a=comp_skeleton_k, b=comp_skeleton)


    if width < 3 or width > 34 or length < 90:
        continue

    if num_of_ends <= 4 and num_of_junctions <= 5 and num_of_ends >= 0:
        # it is a rope-like object
        rope_like_skeletons.append(k)
        rope_like_length.append(np.sum(comp_skeleton_k > 0))

    #dlo_ends = cv2.imread('1_thinned_ends.png')[:, :, 0]
    # dlo_conjunctions = np.array(junctionsImage_k)[:, :, 0] 
    # dlo_conjunctions_dilate = cv2.dilate(dlo_conjunctions, np.ones((5,5), np.uint8), iterations=1)

# only keep the compoents belonging to rope-like objects
rope_like_mask = np.zeros_like(comp_mask)
for k in rope_like_skeletons:
    rope_like_mask[skeleton_labels == k] = 1
    # random pick one pixel from skeleton_labels which is 1
    cord_x = np.where(skeleton_labels == k)[0][0]
    cord_y = np.where(skeleton_labels == k)[1][0]
    #print(np.array(np.where(skeleton_labels == k)).shape, cordinates.shape)
    # find its corresponding pixel in all_labels
    for m in range(num_of_comp):
        label_m = all_labels[cord_x, cord_y]
        #print(k, m, cordinates, label_m)
        if label_m != 0:
            rope_like_mask[all_labels == label_m] = 1
            break

visualize(mm = mask ,a=mask, b=rope_like_mask, c=all_labels, d=comp_skeleton)
# save the post-processed mask
np.save('rope_like_mask.npy', rope_like_mask)