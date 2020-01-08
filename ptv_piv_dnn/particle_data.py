"""
Taking and normalizing the given particle data.

Author:
    Sabareesh Mamidipaka
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
    
    """
    
    def __init__(self, initial_pos, final_pos, t_initial, t_final):
        """
        
        :param : The initial and final positions of particles as an array, the initial and final 
        time instants as an array, and the radius of cluster
        
        """
        
        ## Storing the given data
        self.initial_pos = initial_pos.astype(np.float32)
        self.final_pos = final_pos.astype(np.float32)
        self.t_initial = np.float32(t_initial)
        self.t_final = np.float32(t_final)
        
        ## Calculate and store the statistics of the data
        self.mean_pos = np.mean(initial_pos,axis=0)

        self.min_pos = np.min(np.concatenate([initial_pos, final_pos],axis=0),axis=0)
        self.max_pos = np.max(np.concatenate([initial_pos, final_pos],axis=0),axis=0)
        self.range = np.max_pos-self.min_pos
        
        self.max_time = np.max(self.t_final)
        self.min_time = np.min(self.t_initial)
        self.t_initial_norm = self.scale_time_data(t_initial).astype(np.float32)
        self.t_final_norm = self.scale_time_data(t_final).astype(np.float32)
        self.time_bound = [np.min(self.t_initial_norm), np.max(self.t_final_norm)]

        ## Normalizing the data
        self.initial_pos_norm = self.scale_pos_data(initial_pos).astype(np.float32)
        self.final_pos_norm = self.scale_pos_data(final_pos).astype(np.float32)
        
    def scale_pos_data(self, array):
        """
        Scale position data.
        
        :param array: any spatial data to be normalized w.r.t the mean and variance of the initial position
        :returns: normalized data
        """
        normalized_data = ((array - self.min_pos) / self.range)
        return normalized_data
    
    def scale_time_data(self, time):
        """
        Scale time data.
        
        :param array: any temporal data to be scaled appropriately w.r.t to the final time
        :returns: normalized data
        """
        normalized_data =   time / self.max_time
        return normalized_data

if __name__ == '__main__':
    # Initialize it with various conditions
    # Do some test
    # Print error messages for debugging
    pass
