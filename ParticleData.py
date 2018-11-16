"""

This contains .

"""

class ParticleData:

    def __init__(self, initial, final, t_initial, t_final, vel_layers, radius):
        self.sess = tf.Session()
        self.initial = initial.astype(np.float32)
        self.final = final.astype(np.float32)
        self.t_initial = t_initial.astype(np.float32)
        self.t_final = t_final.astype(np.float32)

        self.vel_weights, self.vel_biases = initialize_NN(vel_layers)

        self.initial = train_data[:,[0,1]].astype(np.float32)
        self.initial_ch, self.mu_train_pos, self.sigma_train_pos=rescale(initial)
        self.final = train_data[:,[2,3]].astype(np.float32)
        self.final_ch=rescale_test(final, self.mu_train_pos, self.sigma_train_pos)
        self.t_scale = np.max(t_final)
        self.t_initial_ch =t_initial/self.t_scale
        self.t_final_ch = t_final/self.t_scale



