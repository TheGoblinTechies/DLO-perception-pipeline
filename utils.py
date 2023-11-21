import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import colors as mcolors
import colorsys
from depth_estimation import PCA_for_slope_to_x_y_plane
from sklearn.cluster import KMeans


def man_cmap(cmap, value=1.):
    colors = cmap(np.arange(cmap.N))
    hls = np.array([colorsys.rgb_to_hls(*c) for c in colors[:,:3]])
    hls[:,1] *= value
    rgb = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0,1)
    return mcolors.LinearSegmentedColormap.from_list("", rgb)


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

def visualize_jet(**images):
    """PLot images in one row."""
    cmap = plt.cm.get_cmap("jet")
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image,cmap=cmap)
    plt.show()


def cropped_1080P(pct, flatten=False):
    if pct.shape[0] == 1920*1080:
        pct_cropped = np.reshape(pct,(1080,1920,3))
        pct_cropped = pct_cropped[:,420:1500,:]
    else:
        pct_cropped = pct
    if flatten == True:
        pct_cropped = np.reshape(pct_cropped, (1080*1080,3))
    return pct_cropped

def pc_flatten(pct):
    x, y, z = pct.shape
    pct = np.reshape(pct, (x*y,z))
    return pct

def cubic_function(param,x):
    a,b,c,d = param[0],param[1],param[2],param[3]
    return a*x**3 + b*x**2 + c*x + d    

def cubic_function_solve(f_0, f_1, dz_0, dz_1, z_0=0, z_1=1):
    A = np.array([[  z_0**3,   z_0**2, z_0, 1],
                  [  z_1**3,   z_1**2, z_1, 1],
                  [3*z_0**2, 2*z_0**1,   1, 0],
                  [3*z_1**2, 2*z_1**1,   1, 0]])
    B = np.array([f_0, f_1, dz_0, dz_1])

    X_solution = np.linalg.solve(A, B)
    return X_solution

def read_dlo_mask(image_path, mask_path, patch_size, image_count, copy = False, vis = False):
    print(image_count, ' load image: ', image_path)
    dlo = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #dlo = cv2.cvtColor(dlo, cv2.COLOR_BGR2RGB)
    cropped_patch_size = int(patch_size)

    y, x = dlo.shape[0], dlo.shape[1]

    # dlo_cropped = dlo[cropped_patch_size:-cropped_patch_size, cropped_patch_size:-cropped_patch_size, :]
    # mask_cropped = mask[cropped_patch_size:-cropped_patch_size, cropped_patch_size:-cropped_patch_size]
    # if vis == True:
    #     plt.imshow(dlo_cropped)
    #     plt.show()
    # dlo_padding = dlo * 0
    # mask_padding = mask * 0
    # dlo_padding[cropped_patch_size:-cropped_patch_size,cropped_patch_size:-cropped_patch_size, :] = dlo_cropped
    # mask_padding[cropped_patch_size:-cropped_patch_size,cropped_patch_size:-cropped_patch_size] = mask_cropped
    
    if not copy:
        dlo_padding  = np.zeros((y + 2 * cropped_patch_size, x + 2 * cropped_patch_size, 3)).astype(np.uint8)
        mask_padding = np.zeros((y + 2 * cropped_patch_size, x + 2 * cropped_patch_size)).astype(np.uint8)
        
        dlo_padding[cropped_patch_size:-cropped_patch_size,cropped_patch_size:-cropped_patch_size, :] = dlo
        mask_padding[cropped_patch_size:-cropped_patch_size,cropped_patch_size:-cropped_patch_size] = mask

        dlo = dlo_padding
        mask = mask_padding
        
    if vis == True:
        plt.imshow(dlo)
        plt.show()
    return dlo, mask