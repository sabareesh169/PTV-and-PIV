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
        self.t_initial = np.float32(t_initial)
        self.t_final = np.float32(t_final)
        
        ## storing the statistics of the training data
        self.mean_pos = np.mean(initial_pos,axis=0)
        self.sigma_pos = np.std(initial_pos,axis=0)

        ## Normalizing the training data
        self.max_time = self.t_final
        self.t_initial_norm = self.rescale_time_data(t_initial, self.initial_pos)
        self.t_final_norm = self.rescale_time_data(t_final, self.initial_pos)

        self.initial_pos_norm = self.rescale_pos_data(initial_pos)
        self.final_pos_norm = self.rescale_pos_data(final_pos)

        ## Restricting the possible matches for each particle in the initial frame.
        tree=spatial.KDTree(initial_pos)
        self.cluster=tree.query_ball_point(initial_pos, radius)
        
        self.time_bound=[np.min(self.t_initial), np.max(self.t_final)]
        
    def rescale_pos_data(self, array):
        """
        
        :param array: any spatial data to be normalized w.r.t the mean and variance of the initial position
        :returns: normalized data
        """
        normalized_data = ((array - self.mean_pos)/ self.sigma_pos).astype(np.float32)
        return normalized_data
    
    def rescale_time_data(self, time, position):
        """
        :param array: any temporal data is converted into appropriate dimensions and scaled appropriately
                      w.r.t to the final time
        :returns: normalized data
        """
        normalized_time =  (np.ones((position.shape[0],1))*time/ self.max_time).astype(np.float32)
        return normalized_time
