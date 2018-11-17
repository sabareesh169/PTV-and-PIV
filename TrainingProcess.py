def sampling_points(initial, final, pred, list_, sigma):
    index = np.arange(initial.shape[0])
    for i in range(initial.shape[0]):
        points_index = np.array(list_[i])
        points = final[points_index]
        p = np.exp(-np.sum((pred[i,:]-points)**2, axis=1)/(2*sigma**2))/(2*sigma)
        cum_sum_q=np.exp(np.log(p)-sc.misc.logsumexp(np.log(p))).cumsum()
        u=np.random.rand()
        idx=np.arange(len(cum_sum_q))
        index[i]=points_index[idx[u<cum_sum_q][0]]
    return index

def optimize_theta(self, true_vel_ch):
    for i in range(50):
        self.sess.run(self.optimizer_phy)
        self.sess.run(self.optimizer_vel, feed_dict={self.VelocityModel.vel_sample: true_vel_ch})

def sampling_theta(self, optimizer_vel, initial, final_index, t_initial, t_final):
    initial_norm = self.ParticleData.rescale_pos_data(initial)
    final_index_norm = self.ParticleData.rescale_pos_data(final_index)
    t_final_norm = self.ParticleData.rescale_time_data(t_final, final_index_norm)
    t_initial_norm = self.ParticleData.rescale_time_data(t_initial, initial_norm)

    true_vel_ch = (final_index_norm-initial_norm)/(t_final_norm-t_initial)
    optimize_theta(self, true_vel_ch)

def sampling_sigma(self, optimizer_sigma, pred, final_index):
    for i in range(50):
        self.sess.run(optimizer_sigma, feed_dict={self.VelocityModel.likelihood : np.sum((pred-final_index)**2)})
          
class TrainingProcess:
    
    """
    Takes in the loss functions constructed in the Velocity Model.
    Runs an optimizer to minimize the loss function
    """
    
    def __init__(self, ParticleData, VelocityModel):
        self.sess = tf.Session()

        self.ParticleData = ParticleData
        self.VelocityModel = VelocityModel
        self.optimizer_phy = tf.train.AdamOptimizer().minimize(VelocityModel.total_residue, var_list=VelocityModel.vel_weights+VelocityModel.vel_biases)
        self.optimizer_vel = tf.train.AdamOptimizer().minimize(VelocityModel.loss_vel, var_list=VelocityModel.vel_weights+VelocityModel.vel_biases)
        self.optimizer_sigma = tf.train.AdamOptimizer().minimize(VelocityModel.neg_log_prob, var_list=VelocityModel.sigma)
    
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    def match_particles_and_velocity(self, n_iter):
        """
        :param size: Sampling the points, theta and sigma one after the other for n_iter number of times.
        :returns: optimized weights of the velocity DNN and the matched particles.
        """
        for i in range(n_iter):
            index=sampling_points(self.ParticleData.initial_pos, self.ParticleData.final_pos, self.sess.run(self.VelocityModel.pos_NN), self.ParticleData.cluster, self.sess.run(self.VelocityModel.sigma))
            final_index=self.ParticleData.rescale_pos_data(self.ParticleData.final_pos[index])
            sampling_theta(self, self.optimizer_vel, self.ParticleData.initial_pos, final_index, self.ParticleData.t_initial, self.ParticleData.t_final)
            sampling_sigma(self, self.optimizer_sigma, self.sess.run(self.VelocityModel.pos_NN), final_index)
        self.index=index  

    def train_velocity(self, n_iter):
        """
        :param size: Sampling the theta and sigma one after the other for n_iter number of times.
        :returns: optimized weights of the velocity DNN.
        """
        for i in range(n_iter):
            sampling_theta(self, self.optimizer_vel, self.ParticleData.initial_pos, self.ParticleData.final_pos, self.ParticleData.t_initial, self.ParticleData.t_final)
            sampling_sigma(self, self.optimizer_sigma, self.sess.run(self.VelocityModel.pos_NN), self.ParticleData.final_pos)

    def vel_predict(self, t, x, y):
        """
        :param : the time and position of the point.
        :returns: velocity at the particular position and time.
        """
        return self.sess.run(self.VelocityModel.vel_predict(t,x,y))                   

    def pos_predict(self, t1, t2, x, y):
        """
        :param : the time and position of the point and the time at whoch the position is required.
        :returns: predicted position of the particle.
        """
        return self.sess.run(self.VelocityModel.pos_predict(t1, t2, x, y))  
