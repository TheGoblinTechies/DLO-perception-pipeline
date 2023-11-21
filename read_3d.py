from itertools import chain
import numpy as np
import open3d as o3d
import cv2
from scipy import io as sio
import matplotlib.pyplot as plt
from depth_estimation import PCA_for_slope_to_x_y_plane
from sklearn.cluster import KMeans
from utils import *
import scipy

def find_point(pts, r, c, width):  # row and col in the 2d image
    index = r * width + c
    x = pts[index][0]
    y = pts[index][1]
    z = pts[index][2]
    return x,y,z

def find_points_from_mask(pts, mask, width):
    non_zero_ele = np.where(mask==1)
    non_zero_index = non_zero_ele[0]*width + non_zero_ele[1]
    non_zero_xyz = pts[non_zero_index] 
    return non_zero_xyz
    
def cubic_function(param,x):
    a,b,c,d = param[0],param[1],param[2],param[3]
    return a*x**3 + b*x**2 + c*x + d


def cd_3d(pct1, pct2):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pct1)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pct2)

    cd_3d_1_to_2 = pcd2.compute_point_cloud_distance(pcd1)
    cd_3d_2_to_1 = pcd1.compute_point_cloud_distance(pcd2)

    chamfer3d_dist = np.mean(cd_3d_1_to_2) + np.mean(cd_3d_2_to_1) 
    return chamfer3d_dist


def interpolation_in_3d(kp_list,density=50):
    kp_interpolated_list = []
    for i in range(len(kp_list)-1):
        interpolated = np.zeros((density,3))
        for j in range(3):
            interpolated[:,j] = np.linspace(kp_list[i][j],kp_list[i+1][j],density, endpoint=False)
        kp_interpolated_list.append(interpolated)
    print('kp_interpolated_list shape:', np.array(kp_interpolated_list).shape)

    return np.array(kp_interpolated_list)

def chain_struct_to_points(chain_struct_3d):
    # input: chain_struct for the whole DLO
    # output: not-in-ordered points for 3d visualization
    pc_list = []
    for i in range(0, len(chain_struct_3d)):
    # loop each point in the chain
        for j in range(0, len(chain_struct_3d[i].chain_pieces_3d)):
            pc_list.append(list(chain_struct_3d[i].chain_pieces_3d[j]))
    return pc_list

def visualize_3d(pct, pct_gt = None, overlapping = False):
    point_cloud = o3d.geometry.PointCloud()
    pct = np.reshape(pct, (-1 , 3))
    if overlapping:
        pct_gt = np.reshape(pct_gt, (-1 , 3))
        pct = np.concatenate((pct, pct_gt), axis=0)
    point_cloud.points = o3d.utility.Vector3dVector(pct)
    #point_cloud.points = o3d.utility.Vector3dVector(pct_cropped)
    o3d.visualization.draw_geometries([point_cloud])

def extract_3dpos_from_chain(chain_struct_2dpx, mask_2d, point_cloud, manual_mask = None, vis_global = True):
    # copy the chain struct 2d px list
    chain_struct_3d = chain_struct_2dpx.copy()

    # pct: point_cloud
    pct = point_cloud
    print('pct.shape: ', pct.shape)
    # 1080 * 1080 * 3
    pct_cropped = cropped_1080P(pct)
    if vis_global:
        visualize_3d(pct_cropped)    
    if manual_mask is not None:
        pct_cropped = pct_cropped * np.stack([manual_mask,manual_mask,manual_mask],axis=2)

    # get valid area
    mask_2d = np.where(mask_2d > 0, 1, 0)
    #visualize_3d(pct_cropped)
    mask_2d_3c = np.stack([mask_2d,mask_2d,mask_2d],axis=2)
    #visualize_3d(pct_cropped)
    pct_cropped = pct_cropped * mask_2d_3c
    pct_cropped_flatten = pc_flatten(pct_cropped)
    
    # include mis-alignment PC
    
    # elimate all distance outside the range
    for i in range(1080):
        for j in range(1080):
            
            # for rubber rod a
            if pct_cropped[i,j,2] < 200 or pct_cropped[i,j,2] > 800: 
                #if pct_cropped[i,j,2] < 200 or pct_cropped[i,j,2] > 800: 
                pct_cropped[i,j,:] = 0

    #               c=pct_cropped[:,:,2],
    #               d=mask_2d,
    #               )
    
    # total num of PC of non zero part, by adding three coordinates together
    pct_cropped_sum = np.sum(pct_cropped,axis=2)
    # get non zero index
    pct_cropped_non_zero = np.array(np.nonzero(pct_cropped_sum)).transpose()
    print('shape pct_cropped_non_zero', pct_cropped_non_zero.shape)
    print(pct_cropped_non_zero)
    #print(pct_cropped_non_zero)
    #pct_cropped_non_zero = np.argwhere(pct_cropped != np.array([0,0,0]))
    
    # flat all 2d px as cluster centers
    all_kp_2dpx_copy = []
    kp_cnt = 0
    for i in range(len(chain_struct_2dpx)):
        all_kp_2dpx_copy.append(chain_struct_2dpx[i].chain_pieces)
        print(i, '-th', len(chain_struct_2dpx[i].chain_pieces))
        kp_cnt += len(chain_struct_2dpx[i].chain_pieces)
        #i_th_chain.chain_pieces_3d = 
    all_kp_2dpx = [list(item) for sublist in all_kp_2dpx_copy for item in sublist]
    print('all_kp_2dpx: ', all_kp_2dpx)
    
    # aligned 
    #####
    # kmeans to allocate mean values
    #####
    
    print('len all_kp_2dpx: ', len(all_kp_2dpx), kp_cnt)
    num_kp = len(all_kp_2dpx)
    # create kmeans
    kmeans = KMeans(n_clusters=num_kp)
    # fix cluster center
    kmeans.fit(all_kp_2dpx)
    print('kmeans.labels_: ', kmeans.labels_)
    # get cluster center index for each p in pc
    pct_cluster_index = kmeans.predict(pct_cropped_non_zero)
    pct_cluster_center = []
    #pct_cropped_non_zero_array = np.array(pct_cropped_non_zero)


    cnt_list = []
    last_pct_center_med = [0,0,0]
    for i in range(num_kp):
        cnt = 0
        pct_center = []
        for j in range(len(pct_cropped_non_zero)):
            if pct_cluster_index[j] == kmeans.labels_[i]:
                x, y = pct_cropped_non_zero[j]
                pct_center.append(list(pct_cropped[x][y]))
                #pct_center += pct_cropped[x][y]
                cnt += 1
        # too few valid points for the kp
        if cnt < 10:
            #print('cal mean: ', i, cnt, pct_center)
            pct_cluster_center.append([0,0,0])
        else:
            #dz_0, z_0, pca = PCA_for_slope_to_x_y_plane(np.array(pct_center))
            # mean is not good enough
            # there might be some outliers due to waterfall effect in the depth map

            # Sort the points by their z-coordinate (third axis)
            pct_center = np.array(pct_center)
            sorted_points = pct_center[np.argsort(pct_center[:, 2])]
            # Choose the 25% closest points based on their z-coordinate
            num_closest_points = int(len(pct_center) * 0.25)
            closest_points = sorted_points[:num_closest_points]
            pct_center_med = np.median(np.array(closest_points), axis=0)
            print('cal mean: ', i, cnt, pct_center_med, np.sum(np.square(last_pct_center_med-pct_center_med)), 
                  np.sum(np.square(last_pct_center_med[-1]-pct_center_med[-1])),
                  np.sum(np.square(last_pct_center_med[:2]-pct_center_med[:2])))
            last_pct_center_med = pct_center_med
            # pct_cluster_center is center position for i-th kp/cluster center
            pct_cluster_center.append(pct_center_med)
            cnt_list.append(cnt)
            
    # plt.plot(dz_list/np.mean(dz_list))
    # plt.plot(cnt_list/np.mean(cnt_list))
    # plt.show()

    #ordered_pct_cluster_center = pct_cluster_center#[x for _, x in sorted(zip(kmeans.labels_, pct_cluster_center))]
    print('ordered_pct_cluster_center:', pct_cluster_center)
    print(len(pct_cluster_center))
    print(pct_cluster_center)

    # get 3d point centers (not evenly distributed in 3d) from 2d kp (almost evenly distributed in 2d)
    pct_cluster_center = np.array(pct_cluster_center)
    if vis_global:
        visualize_3d(pct_cluster_center)

    
    
    
    # plot if in order 

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')

    # chain_struct_2dpx: list, each contain a chain
    # chain_struct_3d: list, each contain a chain
    cnt = 0
    for chain_i in range(len(chain_struct_3d)):
        # print chain struct 3d length
        print(chain_i, 'len(chain_struct_3d[chain_i].chain_pieces: ', len(chain_struct_3d[chain_i].chain_pieces))
        prev_center_med = pct_cluster_center[cnt]
        abondant = -1
        for point_j in range(len(chain_struct_3d[chain_i].chain_pieces)):
            # this is a [u, v] for a kp in a chain
            #kp_2d_uv = chain_struct_2dpx[chain_i].chain_pieces[point_j]
            
            # find its center index in flated array
            # if np.sum(pct_cluster_center[cnt]) == 0:
            #     cnt += 1
            #     continue
            
            print('append point to chain mean: ', point_j, abondant, pct_cluster_center[cnt], prev_center_med, 
                  np.sum(np.square(pct_cluster_center[cnt]    -prev_center_med)), 
                  np.sum(np.square(pct_cluster_center[cnt][-1]-prev_center_med[-1])),
                  np.sum(np.square(pct_cluster_center[cnt][:2]-prev_center_med[:2])))
            # if np.sum(np.square(pct_cluster_center[cnt]    -prev_center_med)) > 500:
            #     abondant *= -1 
            if abondant == -1:
                chain_struct_3d[chain_i].chain_pieces_3d.append(pct_cluster_center[cnt])
            cnt += 1
            if cnt == pct_cluster_center.shape[0] - 1:
                break
            prev_center_med = pct_cluster_center[cnt-1]
            # ax.scatter(pct_cluster_center[kp_i][0], pct_cluster_center[kp_i][1], pct_cluster_center[kp_i][2], marker='o')
            # if cnt % 20 == 19:
            #     plt.show()

        
    cnt = 0 
    for chain_i in range(len(chain_struct_3d)):
        for point_j in range(len(chain_struct_3d[chain_i].chain_pieces_3d)):
            print(chain_i, point_j, cnt, chain_struct_3d[chain_i].chain_pieces_3d[point_j])
            #print(ordered_pct_cluster_center[cnt])
            cnt += 1
    # for chain_i in range(len(chain_struct_3d)):
    #     # delete if too short
    #     if len(chain_struct_3d[chain_i].chain_pieces) < 4:
    #         del chain_struct_3d[chain_i]
    
    return chain_struct_3d, pct_cropped_flatten

