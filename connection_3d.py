from morph_3d import *
from utils import *
from read_3d import chain_struct_to_points, visualize_3d
from depth_estimation import angle_between
from copy import deepcopy
import scipy

# get in-ordered pixels from 2D mask
def connected_compoent(image, ep_0, ep_1):
    # ep_0 to start
    # ep_1 to end
    position = ep_0
    last_direction = -1
    traced_index = [ep_0]
    while not (position[0] == ep_1[0] and position[1] == ep_1[1]):
        #print(last_direction)
        kernel3x3 = image[position[0]-1:position[0]+2, position[1]-1:position[1]+2]
        reorder_kernel = kernel_reorder(kernel3x3, last_direction)
        
        # first 1 3 5 7
        # then  0 2 4 6
        wdxa_dir = np.logical_and(reorder_kernel==1, wdxa)
        if True in wdxa_dir:
            last_direction = np.nonzero(wdxa_dir==True)[0][0]
        elif not True in wdxa_dir:
            qecz_dir = np.logical_and(reorder_kernel==1, qecz)
            last_direction = np.nonzero(qecz_dir==True)[0][0]

        #print(kernel3x3, position, np.nonzero(reorder_kernel==1),last_direction)
        #last_direction = np.nonzero(reorder_kernel==1)[0]

        image[position[0],position[1]] = 0
        position += moving_direction[last_direction]
        traced_index.append(np.copy(position))

    return traced_index


class chain2d:
    def __init__(self, chain_pieces):
        self.chain_pieces = chain_pieces
        self.chain_pieces_3d = []
        self.chain_pieces_3d_edge = []
        self.chain_pieces_3d_edge_percent = []
        self.chain_pieces_3d_edge_accumu = []
        self.total_length = 0
        self.head_tail_tolerance = 4
        # visible 1 for original
        # 0 for reconstructed
        self.visible = []
        if not chain_pieces == []:
            self.get_head_and_tail()

        #print(chain_pieces)

    def get_head_and_tail(self):
        # outsider is placed at first
        # 0 1
        self.head = np.array([self.chain_pieces[0], self.chain_pieces[0 + self.head_tail_tolerance]])
        # -1 -2
        self.tail = np.array([self.chain_pieces[-1], self.chain_pieces[-1 - self.head_tail_tolerance]])

    def get_head_and_tail_3d(self):
        # outsider is placed at first
        # # 0 1
        # self.head = np.array([self.chain_pieces_3d[0], self.chain_pieces_3d[0 + self.head_tail_tolerance]])
        # # -1 -2
        # self.tail = np.array([self.chain_pieces_3d[-1], self.chain_pieces_3d[-1 - self.head_tail_tolerance]])

        self.head = np.array([self.chain_pieces_3d[0],  np.mean(self.chain_pieces_3d[0 : 0 + self.head_tail_tolerance], axis=0)])
        self.tail = np.array([self.chain_pieces_3d[-1], np.mean(self.chain_pieces_3d[-1 - self.head_tail_tolerance : -1], axis=0)])

class chain_connection:
    def __init__(self, index):
        self.index = index
        self.h_from_index = -1 # start
        self.h_from_h_or_t = 0
        self.t_to_index = -1 # end
        self.t_to_h_or_t = 0
        self.direction = 0 # 0 for regular, 1 for inverse
        self.chain_pieces_3d = None
    
    def reverse(self):
        self.h_from_index, self.t_to_index = self.t_to_index, self.h_from_index
        self.h_from_h_or_t, self.t_to_h_or_t = self.t_to_h_or_t, self.h_from_h_or_t
        #print('self.chain_pieces: ', self.chain_pieces, self.chain_pieces)
        self.chain_pieces_3d = np.flip(self.chain_pieces_3d, 0)
        self.chain_pieces_3d = list(self.chain_pieces_3d)

def connect_with_bezier(chain0ht, chain1ht):
    # two chain, chain0 and chain1, head or tail 0 and 1
    #s0 = np.array([chain0ht[1],chain0ht[0]])
    s0 = chain0ht
    s1 = chain1ht
    #s1 = np.array([chain1ht[1],chain1ht[0]])
    return create_bezier3d(s0, s1)



def merging_cost_keipour(s1_pos, s2_pos):
    s1 = s1_pos[0] - s1_pos[1]
    s2 = s2_pos[0] - s2_pos[1]
    
    s1_norm = np.linalg.norm(s1)
    s2_norm = np.linalg.norm(s2)
        
    s21 = s2_pos[0] - s1_pos[0]
    s12 = s1_pos[0] - s2_pos[0]
   
    s21_norm = np.linalg.norm(s21)
    s12_norm = np.linalg.norm(s12)
    
    euclidean_cost = np.sum(np.square(s1_pos[0] - s2_pos[0]))
    direction_cost = np.abs(np.arccos(-np.dot(s1,s2)/(s1_norm*s2_norm)))
    curveture_cost1 = np.abs(np.arccos(-np.dot(s1,s21)/(s1_norm*s21_norm)))
    curveture_cost2 = np.abs(np.arccos(-np.dot(s2,s12)/(s2_norm*s12_norm)))
    #print('euclidean_cost, direction_cost',euclidean_cost, direction_cost)
    #print('curveture_cost1,curveture_cost1',curveture_cost1,curveture_cost1)
    curveture_cost = np.maximum(curveture_cost1, curveture_cost2)
    if np.isnan(direction_cost):
        direction_cost = 1e3
    a, b, c = 1, 1700, 1700
    total_cost = a * euclidean_cost + b * direction_cost + c * curveture_cost
    return a * euclidean_cost, b * direction_cost, c * curveture_cost, total_cost

# # total interpolated spline energy
# def merging_cost_3d(spline_kp, spline_length):

#     num_nodes = spline_kp.shape[0]
#     k_l = 1
#     k_b = 100
#     total_length = spline_length * k_l
#     total_bending = 0
#     for i in range(num_nodes - 2):
#         v1 = spline_kp[i+1] - spline_kp[i]
#         v2 = spline_kp[i+2] - spline_kp[i+1]
#         total_bending += angle_between(v1, v2)**2 * k_b

    return total_bending + total_length

def create_bezier3d(s1, s2, ls=3):
    # vector s1 and s2
    # s point with 3-dim
    # control point in far away

    s1_vec = s1[0] - s1[1] 
    s2_vec = s2[0] - s2[1]

    s12_norm = np.linalg.norm(s1[0] - s2[0])

    s1_dir = s1_vec / np.linalg.norm(s1_vec)
    s2_dir = s2_vec / np.linalg.norm(s2_vec)

    factor = 0.5
    # extented control points
    c1 = s1[0] + s1_dir * s12_norm * factor
    c2 = s2[0] + s2_dir * s12_norm * factor

    nodes = np.asfortranarray([[s1[0][0], c1[0], c2[0], s2[0][0]],
                               [s1[0][1], c1[1], c2[1], s2[0][1]],
                               [s1[0][2], c1[2], c2[2], s2[0][2]],])

    curve = bezier.Curve(nodes, degree=3)

    sample_number = int(curve.length)#(curve.length // ls) * ls
    
    print('sample_number: ', sample_number)
    
    point = []
    kp = []
    for i in range(0, sample_number + 1):
        point.append(curve.evaluate(1 / sample_number * i))
    for i in range(0, sample_number + 1, int(ls*2.5)):
        kp.append(curve.evaluate(1 / sample_number * i))
    # for bspline curve
    #return np.array(point[0:2]), np.array(kp[0:2])   
    # for bezier curve
    return np.array(point[1:-1]), np.array(kp[1:-1])

def connectivity_explore(chain_struct_3d, point_cloud_path, vis_global=True):
    
    # however, two ends of the DLO pieces have the bias

    for i in range(0, len(chain_struct_3d)):
        
        point_cnt = 0
        # key point have zero center, remove
        while True:
            if np.sum(chain_struct_3d[i].chain_pieces_3d[point_cnt]) == 0:
                del chain_struct_3d[i].chain_pieces_3d[point_cnt] 
                point_cnt -= 1
            # end loop
            if point_cnt == len(chain_struct_3d[i].chain_pieces_3d) - 1:
                break
            
            point_cnt += 1

    # remove two ends (4 points for now)
    discard_num_end_point = 3
    #chain_struct_3d[i].chain_pieces_3d = chain_struct_3d[i].chain_pieces_3d[discard_num_end_point:-discard_num_end_point]

    
    chain_struct_3d_d = []
    for i in range(0, len(chain_struct_3d)):
        chain_struct_3d_d.append(deepcopy(chain_struct_3d[i]))
        chain_struct_3d_d[-1].chain_pieces_3d = chain_struct_3d_d[-1].chain_pieces_3d[discard_num_end_point:-discard_num_end_point]
    if vis_global:
        
        visualize_3d(chain_struct_to_points(chain_struct_3d_d))
    # copy back
    chain_struct_3d = chain_struct_3d_d
    
    # too few
    cnt = 0
    
    while True:
        print(cnt, len(chain_struct_3d))
        print('chain_struct_3d: ', chain_struct_3d[cnt])
        if chain_struct_3d[cnt].chain_pieces_3d == [] or len(chain_struct_3d[cnt].chain_pieces_3d) < 5:
            del chain_struct_3d[cnt]
            if cnt >= len(chain_struct_3d) - 1:
                break
            cnt -= 1
        if cnt >= len(chain_struct_3d) - 1:
            break
        cnt += 1


    for i in range(0, len(chain_struct_3d)):
        chain_struct_3d[i].get_head_and_tail_3d()

    zero_map = np.zeros((1500, 1500, 3))
    # loop each chain
    for i in range(0, len(chain_struct_3d)):
        # loop each point in the chain
        for j in range(0, len(chain_struct_3d[i].chain_pieces_3d)):
            zero_map[int(chain_struct_3d[i].chain_pieces_3d[j][0]+300),
                     int(chain_struct_3d[i].chain_pieces_3d[j][1]+300),
                     :] = 1
                     #int(chain_struct_3d[i].chain_pieces[j][2]) ] = 1
            # if j % 5 == 4:
            #     visualize(a = zero_map[:,:])

    pc_list = chain_struct_to_points(chain_struct_3d)
    print('pc_list: ', pc_list)
    #visualize_3d(np.array(pc_list))

    total_merge = np.ones((len(chain_struct_3d), len(chain_struct_3d), 4)) * 10e7
    for j in range(0, len(chain_struct_3d)):
        for k in range(j+1, len(chain_struct_3d)):
            total_merge[j,k,0] = merging_cost_keipour(chain_struct_3d[j].head, chain_struct_3d[k].head)[-1]
            total_merge[j,k,1] = merging_cost_keipour(chain_struct_3d[j].tail, chain_struct_3d[k].head)[-1]
            total_merge[j,k,2] = merging_cost_keipour(chain_struct_3d[j].head, chain_struct_3d[k].tail)[-1]
            total_merge[j,k,3] = merging_cost_keipour(chain_struct_3d[j].tail, chain_struct_3d[k].tail)[-1]

            print(j,' ',k,' h to h ', total_merge[j,k,0])
            print(j,' ',k,' t to h ', total_merge[j,k,1])
            print(j,' ',k,' h to t ', total_merge[j,k,2])
            print(j,' ',k,' t to t ', total_merge[j,k,3])
            
    new_piece = []
    chain_connection_list = []

    # load visible chain from chain_struct_3d
    for i in range(len(chain_struct_3d)):
        chain_connection_list.append(chain_connection(i))
        chain_connection_list[-1].chain_pieces_3d = chain_struct_3d[i].chain_pieces_3d

    chain_cnt = len(chain_struct_3d)
    #print('chain_connection_list: ', len(chain_connection_list), chain_connection_list[0])
    visible_chain_cnt = chain_cnt

    total_merge_flat = total_merge.flatten()
    num_of_chain = len(chain_struct_3d)
    if num_of_chain > 1:
        threshold =  np.partition(total_merge_flat, num_of_chain-2)[num_of_chain-2] + 10
    else:
        threshold = 2e4
    for j in range(len(chain_struct_3d)):
        for k in [0,1]:
            cost_j = total_merge[j,:,:]
            minimal_j = np.min(cost_j)
            minimal_j_idx = np.argmin(cost_j)
            
            connection_type = ['h to h',
                                't to h',
                                'h to t',
                                't to t']
            h_t = ['h', 't']
            
            connect_from = j
            #print(j, k, minimal_j_idx)
            connect_to = minimal_j_idx//4
            connect_from_h_t = (minimal_j_idx%4)%2
            connect_to_h_t = (minimal_j_idx%4)//2

            connection_type = minimal_j_idx % 4

            if minimal_j < threshold:
                # a valid connection
                # set j head or tail to very large
                total_merge[j,:,[connect_from_h_t,connect_from_h_t+2]] = 10e8
                total_merge[:,connect_to,[connect_to_h_t*2,connect_to_h_t*2+1]] = 10e8
                # set connect to head or tail to very large
                total_merge[connect_to,:,[connect_to_h_t,connect_to_h_t+2]] = 10e8

                print(connect_from, ' ', h_t[connect_from_h_t] , ' connect to ', connect_to, ' ', h_t[connect_to_h_t], ' and chain_cnt is ', chain_cnt)
                
                chain_connection_list.append(chain_connection(chain_cnt))
                
                if h_t[connect_from_h_t] == 'h':
                    # h to h
                    chain_connection_list[connect_from].h_from_index = chain_cnt
                    chain_connection_list[connect_from].h_from_h_or_t = 'h'
                else:
                    # t to h
                    chain_connection_list[connect_from].t_to_index = chain_cnt
                    chain_connection_list[connect_from].t_to_h_or_t = 'h'

                if h_t[connect_to_h_t] == 'h':
                    chain_connection_list[connect_to].h_from_index = chain_cnt
                    chain_connection_list[connect_to].h_from_h_or_t = 't'
                else:
                    chain_connection_list[connect_to].t_to_index = chain_cnt
                    chain_connection_list[connect_to].t_to_h_or_t = 't'
                
                chain_connection_list[connect_to].chain_pieces_3d = chain_struct_3d[connect_to].chain_pieces_3d
                chain_connection_list[connect_from].chain_pieces_3d = chain_struct_3d[connect_from].chain_pieces_3d

                chain_connection_list[-1].h_from_index = connect_from
                chain_connection_list[-1].h_from_h_or_t = h_t[connect_from_h_t]
                chain_connection_list[-1].t_to_index = connect_to
                chain_connection_list[-1].t_to_h_or_t = h_t[connect_to_h_t]
                
                
                chain0 = chain_struct_3d[connect_from].head if h_t[connect_from_h_t] == 'h' else chain_struct_3d[connect_from].tail
                chain1 = chain_struct_3d[connect_to].head   if h_t[connect_to_h_t]   == 'h' else chain_struct_3d[connect_to].tail
                
                # todo add a function to connect two chain
                connected_piece, connected_kp = connect_with_bezier(chain0, chain1)
                
                new_piece.append(connected_piece)
                #print('new piece append connected_piece', connected_piece.shape, connected_piece)
                chain_connection_list[-1].chain_pieces_3d = list(np.array(connected_kp)[:,:,0])
                #print('chain0, chain1: ', chain0, chain1)
                #chain_connection_list[-1].get_head_and_tail_3d()
                #print('new added: ', cnt, 'h from ', chain_connection_list[-1].h_from_index, 'with h and t to ', chain_connection_list[-1].t_to_index,  'head: ',chain_connection_list[-1].head, 'and tail: ', chain_connection_list[-1].tail, '. However, the h from chain ')
                chain_cnt += 1

    # for loop to merge every chain
    # two different chains: visible and reconstructed
    #connection_threshold = 50
    full_chain = chain2d([])
    full_chain_inside = []
    for i in range(len(chain_connection_list)):
        print(chain_connection_list[i].index, np.array(chain_connection_list[i].chain_pieces_3d).shape, 
                chain_connection_list[i].h_from_index,
                chain_connection_list[i].h_from_h_or_t,
                chain_connection_list[i].t_to_index,
                chain_connection_list[i].t_to_h_or_t)
        
        if chain_connection_list[i].h_from_index == -1:
            # start from one end (head)
            full_chain.chain_pieces_3d += list(chain_connection_list[i].chain_pieces_3d)
            full_chain.visible += [1 for i in range(len(full_chain.chain_pieces_3d))]
            full_chain_inside = [i]
            break
        elif chain_connection_list[i].t_to_index == -1:
            # start from one end (tail)
            full_chain.chain_pieces_3d += list(chain_connection_list[i].chain_pieces_3d)[::-1]
            full_chain.visible += [1 for i in range(len(full_chain.chain_pieces_3d))]
            full_chain_inside = [i]
            break

    # loop over all chain
    connect_cnt = 0
    while len(full_chain_inside) != len(chain_connection_list):
        # full_chain_inside[-1] is the to-connect chain
        print('full_chain_inside and threshold', len(full_chain_inside), threshold)
        print('chain_connection_list len', len(chain_connection_list))
        current_h_from_index = chain_connection_list[full_chain_inside[-1]].h_from_index
        current_h_from_h_or_t = chain_connection_list[full_chain_inside[-1]].h_from_h_or_t
        current_t_to_index = chain_connection_list[full_chain_inside[-1]].t_to_index
        current_t_to_h_or_t = chain_connection_list[full_chain_inside[-1]].t_to_h_or_t
        #print('list(chain_connection_list[current_h_from_index].chain_pieces_3d)', list(chain_connection_list[current_h_from_index].chain_pieces_3d))
        print(full_chain_inside[-1], current_h_from_index, current_h_from_h_or_t, current_t_to_index, current_t_to_h_or_t, full_chain_inside)
        connect_cnt += 1
        if connect_cnt == 10:
            break
        if current_h_from_index != -1 and (current_h_from_index not in full_chain_inside):
            if current_h_from_h_or_t == 'h':
                full_chain.chain_pieces_3d += list(chain_connection_list[current_h_from_index].chain_pieces_3d)

            elif current_h_from_h_or_t == 't':
                full_chain.chain_pieces_3d += list(chain_connection_list[current_h_from_index].chain_pieces_3d)[::-1]

            if current_h_from_index >= visible_chain_cnt:
                # reconstructed
                full_chain.visible += [0 for i in range(len(list(chain_connection_list[current_h_from_index].chain_pieces_3d)))]
            else:
                # visible
                full_chain.visible += [1 for i in range(len(list(chain_connection_list[current_h_from_index].chain_pieces_3d)))]

            full_chain_inside.append(current_h_from_index)

        elif current_t_to_index != -1 and (current_t_to_index not in full_chain_inside):

            if current_t_to_h_or_t == 'h':
                full_chain.chain_pieces_3d += list(chain_connection_list[current_t_to_index].chain_pieces_3d)

            elif current_t_to_h_or_t == 't':
                full_chain.chain_pieces_3d += list(chain_connection_list[current_t_to_index].chain_pieces_3d)[::-1]

            if current_t_to_index >= visible_chain_cnt:
                # reconstructed
                full_chain.visible += [0 for i in range(len(list(chain_connection_list[current_t_to_index].chain_pieces_3d)))]
            else:
                # visible
                full_chain.visible += [1 for i in range(len(list(chain_connection_list[current_t_to_index].chain_pieces_3d)))]

            full_chain_inside.append(current_t_to_index)            
        #x = input('w8')
    print('full_chain', len(full_chain.chain_pieces_3d), full_chain_inside)
    
    

    point_cloud = np.load(point_cloud_path)
    pct = chain_struct_to_points([full_chain])
    #pct = chain_struct_to_points(chain_connection_list)
    if vis_global:
        visualize_3d(pct)
        visualize_3d(pct, point_cloud, overlapping=True)

    return full_chain


def keipour_connection(dlo_image_name, manual_mask = None, vis_global=True):
    dlo_dilated, dlo = read_dlo(dlo_image_name, manual_mask, vis=vis_global)
    dlo_thinned = get_skeleton(dlo_dilated,vis=vis_global)
    dlo_thinned[:,[0,-1]] = 0
    dlo_thinned[[0,-1],:] = 0
    dlo_thinned_3 = np.stack([dlo_thinned,dlo_thinned,dlo_thinned], axis=2)
    #cv2.imwrite('1_thinned.png', dlo_thinned_3)

    # Clone the original image as we are about to destroy it
    #endsImage = Image(filename='1_thinned.png')
    endsImage = Image.from_array(dlo_thinned_3)
    endsImage.morphology(method='hit_and_miss', kernel=lineEnds)
    #endsImage.save(filename='1_thinned_ends.png')

    # Clone the original image as we are about to destroy it
    junctionsImage = Image.from_array(dlo_thinned_3)
    junctionsImage.morphology(method='hit_and_miss', kernel=lineJunctions)
    #junctionsImage.save(filename='1_thinned_conjunctions.png')

    #dlo_ends = cv2.imread('1_thinned_ends.png')[:, :, 0]
    dlo_conjunctions = np.array(junctionsImage)[:, :, 0] #cv2.imread('1_thinned_conjunctions.png')[:, :, 0]
    dlo_conjunctions_dilate = cv2.dilate(dlo_conjunctions, kernel_5) 

    dlo_remove_junctions = dlo_thinned - dlo_conjunctions_dilate
    dlo_remove_junctions = np.where(dlo_remove_junctions < 15,  0,  255).astype(np.uint8)
    # plt.imshow(dlo_remove_junctions)
    # plt.show()
    num_labels, labels_im = cv2.connectedComponents(dlo_remove_junctions)
    print(num_labels, labels_im.shape)

    # pixel length
    ls = 10
    chain_struct = []
    empty_map = np.zeros((1088,1088))

    for i in range(1, num_labels):
        chain_map = np.where(labels_im == i * 1.0, 255, 0).astype(np.uint8)
        chain_map_3 = np.stack([chain_map,chain_map,chain_map],axis=2).astype(np.uint8)
        chain_map_image = Image.from_array(chain_map_3)
        chain_map_image.morphology(method='hit_and_miss', kernel=lineEnds)
        chain_map_image = np.array(chain_map_image) 
        chain_map_image[[0,-1],[0,-1],:] = 0
        
        chain_map_display = np.copy(chain_map)
        # get two end points of a chain / connected component
        
        chain_map_ep = np.array(chain_map_image)[:,:,0]
        chain_map_display = np.where(chain_map_display>1,1,0)

        if np.nonzero(chain_map_ep)[0].size == 0:
            continue

        print('chain_map_ep: ', np.nonzero(chain_map_ep)[0], np.nonzero(chain_map_ep)[0] == np.array([], dtype=np.int64))
              
        chain_ep = [[np.nonzero(chain_map_ep)[0][0],np.nonzero(chain_map_ep)[1][0]],
                    [np.nonzero(chain_map_ep)[0][1],np.nonzero(chain_map_ep)[1][1]]]

        order_index = connected_compoent(chain_map//255, chain_ep[0], chain_ep[1])
        
        print(len(order_index), chain_ep, np.nonzero(chain_map_ep))
    
        if len(order_index) < 30:
            continue

        for i in order_index[0::ls][4:-4]:
            y,x = i[0],i[1]
            cv2.circle(empty_map, (x,y), 4, (255,255,255), -1)
            #empty_map[y,x] = 1

        # cv2.imshow('chain_map', empty_map)
        # k = cv2.waitKey(0)
        # # if k == 27:         # wait for ESC key to exit
        #     cv2.destroyAllWindows()
        chain_struct.append(chain2d(order_index[0::ls]))

    mask_2d = dlo
    return chain_struct, mask_2d 


def cal_dist(pA, pB):
    return np.sqrt(np.sum(np.square(pA- pB)))

def cal_points(p0, p1, t):
    alpha = t / cal_dist(p0, p1)
    return p0 + alpha * (p1 - p0)

def reorder_point(chain_struct_3d, t):
    # input 0<=t<=1 
    
    # find the n-th point and n+1 -th point
    #print(t, np.argwhere(t >= chain_struct_3d.chain_pieces_3d_edge_accumu))
    n = np.max(np.argwhere(t >= chain_struct_3d.chain_pieces_3d_edge_accumu))
    #print('larger than:', np.argwhere(t >= chain_struct_3d.chain_pieces_3d_edge_accumu))
    new_point = cal_points(chain_struct_3d.chain_pieces_3d[n], 
               chain_struct_3d.chain_pieces_3d[n+1],
               (t-chain_struct_3d.chain_pieces_3d_edge_accumu[n])*chain_struct_3d.total_length)

    # return a 3d point
    return new_point


def interpolate_reorder(chain_struct_3d, n_new_point = 60, vis_global = True):


    x, y, z = [], [], []
    for i in range(len(chain_struct_3d.chain_pieces_3d)):
        x.append(chain_struct_3d.chain_pieces_3d[i][0])
        y.append(chain_struct_3d.chain_pieces_3d[i][1])
        z.append(chain_struct_3d.chain_pieces_3d[i][2])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    new_points = np.array([x,y,z]).transpose()
    

    unique_points, indices = np.unique(new_points, axis=0, return_index=True)
    sorted_indices = np.sort(indices)
    unique_points = new_points[sorted_indices]
    #unique_elements, index_array = np.unique(unique_points, return_index=True)

    # Sort the unique elements by their first index of occurrence
    #unique_elements_in_original_order = unique_elements[np.argsort(index_array)]
    #unique_points = unique_elements_in_original_order
    #print(unique_elements_in_original_order)
    x = unique_points[:,0]
    y = unique_points[:,1]
    z = unique_points[:,2]

    # B spline

    tck, u = scipy.interpolate.splprep([x,y,z], s = 40)
    new_points = np.array(scipy.interpolate.splev(np.linspace(0, 1, n_new_point), tck))

    # for i in range(0, len(list(x))):
    #     print(x[i], y[i], z[i], new_points[:,i])

    # # have errors here
    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(projection='3d')
    # #ax.set_aspect('equal')
    # # connect the points
    # for i in range(0, len(list(x))-1):
    #     ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], color='b')
    # #ax.scatter(new_points[0,:],new_points[1,:],new_points[2,:] )
    # plt.show()

    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(projection='3d')
    # #ax.set_aspect('equal')
    # ax.scatter(x,y,z)
    # plt.show()

    chain_struct_3d.chain_pieces_3d = list(new_points.transpose())
    if vis_global:
        visualize_3d(chain_struct_to_points([chain_struct_3d]))
    num_points = len(chain_struct_3d.chain_pieces_3d)

    for i in range(num_points-1):
        dist = cal_dist(chain_struct_3d.chain_pieces_3d[i+1],chain_struct_3d.chain_pieces_3d[i])
        chain_struct_3d.chain_pieces_3d_edge.append(dist)

    chain_struct_3d.chain_pieces_3d_edge = np.array(chain_struct_3d.chain_pieces_3d_edge)
    chain_struct_3d.total_length = np.sum(chain_struct_3d.chain_pieces_3d_edge)
    chain_struct_3d.chain_pieces_3d_edge_percent = chain_struct_3d.chain_pieces_3d_edge / chain_struct_3d.total_length
    chain_struct_3d.chain_pieces_3d_edge_accumu = np.add.accumulate(chain_struct_3d.chain_pieces_3d_edge_percent)
    chain_struct_3d.chain_pieces_3d_edge_accumu = list(chain_struct_3d.chain_pieces_3d_edge_accumu )
    chain_struct_3d.chain_pieces_3d_edge_accumu.insert(0,0)
    chain_struct_3d.chain_pieces_3d_edge_accumu = np.array(chain_struct_3d.chain_pieces_3d_edge_accumu)



    # plt.plot(chain_struct_3d.chain_pieces_3d_edge_accumu)
    # plt.show()

    
    new_point_list = []
    for i in range(n_new_point):
        new_point_list.append(reorder_point(chain_struct_3d, i/n_new_point))
        #if i >= 1:
        #    print(i, new_point_list[-1], cal_dist(new_point_list[-1],new_point_list[-2]))

    new_pct = np.array(new_point_list)
    # new_chain = chain2d([])
    # new_chain.chain_pieces_3d = new_point_list
    if vis_global:
        visualize_3d(new_pct)
    # check dulicated points
    # and use scipy to interpolate
    return chain_struct_3d, new_pct
    #for i in range(len(chain_struct_3d)):