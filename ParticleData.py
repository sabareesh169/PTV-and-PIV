"""

This class takes in the particle data and scales the data for better functioning of the DNN's.
The postional data is normalised w.r.t the initial position of the particles. 
The time data is 'normalized' differently to avoid (potentially) division with zero.
The position data is then used to form multiple clusters to limit the possible matches for each initial point.
"""

class ParticleData:
    
    def __init__(self, initial_pos, final_pos, t_initial, t_final, radius):
        """
        
        :param : The initial and final positions of particles, the initial and final time instants, and the radius of cluster
        """        
        self.sess = tf.Session()
        self.initial_pos = initial_pos.astype(np.float32)
        self.final_pos = final_pos.astype(np.float32)
        self.t_initial = np.ones((initial_pos.shape[0],1))*t_initial
        self.t_final = np.ones((final_pos.shape[0],1))*t_final

        self.mean_pos = np.mean(initial_pos)
        self.sigma_pos = np.std(initial_pos)

        self.max_time = t_final
        self.t_initial_norm = self.rescale_time_data(t_initial)
        self.t_final_norm = self.rescale_time_data(t_final)

        self.initial_pos_norm = self.rescale_pos_data(initial_pos)
        self.final_pos_norm = self.rescale_pos_data(final_pos)

        tree=spatial.KDTree(initial)
        self.cluster=tree.query_ball_point(initial, radius)
        self.time_bound=[np.min(self.t_initial), np.max(self.t_final)]
        
    def rescale_pos_data(self, array):
        """
        
        :param array: any spatial data to be normalized w.r.t the mean and variance of the initial position
        :returns: normalized data
        """
        normalized_data = ((array - self.mean_pos)/ self.sigma_pos)
        return rescaled
    
    def rescale_time_data(self, time, position):
        """
        :param array: any temporal data to be scaled appropriately w.r.t to the final time
        :returns: normalized data
        """
        normalized_data =  np.ones((position.shape[0],1))*time/ self.max_time
        return normalized_data
