"""
Taking and normalizing the given particle data.

Author:
    Sabareesh Mamidipaka, Ilias Bilionis
Date:
    11/19/2018
"""

__all__ = ['ParticleData']


# Import everything you need here
import numpy as np
import scipy
from scipy import spatial


class ParticleData(object):
    """
    
    This class takes in the particle data and scales the data for better 
    functioning of the DNN's.
    The postional data is normalised w.r.t the initial position of the particles. 
    The time data is 'normalized' differently to avoid (potentially) division with zero.
    The position data is then used to form multiple clusters to limit the possible 
    matches for each initial point.
    
    """
    
    def __init__(self, initial_pos, final_pos, t_initial, t_final, radius):
        """
        
        :param : The initial and final positions of particles, the initial and final 
        time instants, and the radius of cluster
        """        
        self.sess = tf.Session()
        self.initial_pos = initial_pos.astype(np.float32)
        self.final_pos = final_pos.astype(np.float32)
        self.t_initial = np.float32(t_initial)
        self.t_final = np.float32(t_final)
        
        self.mean_pos = np.mean(initial_pos,axis=0)
        self.sigma_pos = np.std(initial_pos,axis=0)
        self.radius = radius

        self.max_time = self.t_final
        self.t_initial_norm = self.scale_time_data(t_initial, self.initial_pos)
        self.t_final_norm = self.scale_time_data(t_final, self.initial_pos)

        self.initial_pos_norm = self.scale_pos_data(initial_pos)
        self.final_pos_norm = self.scale_pos_data(final_pos)

        tree = spatial.KDTree(initial_pos)
        self.cluster = tree.query_ball_point(initial_pos, radius)
        self.time_bound = [np.min(self.t_initial), np.max(self.t_final)]
        
    def scale_pos_data(self, array):
        """
        Scale position data.
        
        :param array: any spatial data to be normalized w.r.t the mean and variance of the initial position
        :returns: normalized data
        """
        normalized_data = ((array - self.mean_pos) / self.sigma_pos).astype(np.float32)
        return normalized_data
    
    def scale_time_data(self, time, position):
        """
        Scale time data.
        
        :param array: any temporal data to be scaled appropriately w.r.t to the final time
        :returns: normalized data
        """
        normalized_data =  (np.ones((position.shape[0],1)) * time / self.max_time).astype(np.float32)
        return normalized_data


if __name__ == '__main__':
    # Initialize it with various conditions
    # Do some test
    # Print error messages for debugging
    pass
