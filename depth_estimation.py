import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg') # might not be necessary for you
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def PCA_for_slope_to_x_y_plane(X, vis=False):

    pca = PCA(n_components=2)
    pca.fit(X)
    p = pca.components_
    centroid = np.mean(X, 0)
    segments_0 = np.arange(-100, 100)[:, np.newaxis] * p[0]
    segments_1 = np.arange(-100, 100)[:, np.newaxis] * p[1]
    
    norm = np.cross(p[0],p[1])
    if vis:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatterplot = ax.scatter(*(X.T))
        lineplot_0 = ax.plot(*(centroid + segments_0).T, color="red")
        lineplot_1 = ax.plot(*(centroid + segments_1).T, color="blue")
        
        segments_2 = np.arange(-100, 100)[:, np.newaxis] * norm
        lineplot_1 = ax.plot(*(centroid + segments_2).T, color="green")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
   
    vector_0 = np.copy(norm)
    vector_1 = np.copy(norm)
    vector_1[-1] = 0
    angle = angle_between(vector_0,vector_1)
    return angle, centroid, pca
