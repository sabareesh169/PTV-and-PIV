"""

This class takes in the particle data and scales the data for better functioning of the DNN's.
The postional data is normalised w.r.t the initial position of the particles. 
The time data is 'normalized' differently to avoid (potentially) division with zero.

"""

class ParticleData:

    def __init__(self, initial_pos, final_pos, t_initial, t_final, vel_layers, radius):
        self.sess = tf.Session()
        self.initial_pos = initial_pos.astype(np.float32)
        self.final_pos = final_pos.astype(np.float32)
        self.t_initial = t_initial.astype(np.float32)
        self.t_final = t_final.astype(np.float32)

        self.mean_pos = np.mean(initial_pos)
        self.sigma_pos = np.std(initial_pos)

        self.max_time = np.max(t_final)
        self.t_initial_norm = self.rescale_time_data(t_initial)
        self.t_final_norm = self.rescale_time_data(t_final)

        self.initial_pos_norm = self.rescale_pos_data(initial_pos)
        self.final_pos_norm = self.rescale_pos_data(final_pos)
        
    def rescale_pos_data(self, array):
        rescaled = (array - self.mean_pos)/ self.sigma_pos)
        return rescaled
    
    def rescale_time_data(self, array):
        rescaled =  array/ self.max_time
        return rescaled




