from connection_3d import *
from read_3d import *
from scipy import io as sio
import argparse


mask_dir = 'xxx'
print('loading image: ', mask_dir)
chain_struct_2dpx, mask_2d = keipour_connection(mask_dir)
# to do, 1080 * 1080 * 3, with pc[u,v] = [x, y, z]
pc_dir = 'xxx'
point_cloud_path = pc_dir
point_cloud = np.load(point_cloud_path)

#visualize_3d(point_cloud)
chain_struct_3d, pct_cropped_flatten = extract_3dpos_from_chain(chain_struct_2dpx, mask_2d, point_cloud)
chain_struct_3d_connected = connectivity_explore(chain_struct_3d, point_cloud_path)
chain_struct_3d_interpolated, new_pct = interpolate_reorder(chain_struct_3d_connected, vis_global=True)
