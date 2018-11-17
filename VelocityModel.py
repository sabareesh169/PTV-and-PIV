class VelocityModel:
    """
    Takes in the position of the particles at two different time instants.
    Matches the positions of the corresponsing particles.
    Then predicts velocity and 
    """

    def __init__(self, ParticleData, vel_layers, rho, mu, collacation_points=1000):
        
        self.vel_weights, self.vel_biases = initialize_NN(vel_layers)
        self.rho = rho
        self.mu = mu
        self.collacation_points=collacation_points
        self.ParticleData=ParticleData
        
        self.vel_NN = neural_net(ParticleData.t_initial_norm, ParticleData.initial_pos[:,0][:,None], ParticleData.initial_pos[:,1][:,None], self.vel_weights, self.vel_biases)
        self.pos_NN = self.vel_NN*self.ParticleData.sigma_pos*(ParticleData.t_initial- ParticleData.t_final)/self.ParticleData.max_time
        
        self.vel_sample = tf.placeholder(tf.float32, shape=(ParticleData.initial_pos.shape[0], 2))
        self.loss_vel = tf.reduce_sum(tf.square(self.vel_NN[:,:2] - self.vel_sample))
        self.loss_NS_x, self.loss_NS_y, self.loss_cont = self.residue(self.vel_weights, self.vel_biases)
        self.total_residue = self.loss_NS_x + self.loss_NS_y + self.loss_cont
        alpha = tf.constant(0.001, dtype=tf.float32)
        beta = tf.constant(1., dtype=tf.float32)
        self.sigma = tf.Variable(1., dtype=tf.float32)
        self.likelihood = tf.placeholder(dtype=tf.float32)
        self.neg_log_prob = (2*alpha+102)*tf.log(self.sigma)+(beta+self.likelihood/2)/self.sigma**2
        
    def residue(self, vel_weights, vel_biases):
        """
        :param size: parameters of the network, the bounds on space and time for the data.
        :returns: Residue of the governing equations.
        """
        X_c = lhs(2, samples=self.collacation_points, criterion='m').astype(np.float32)
        X_c = (np.asarray((-1.5)+X_c*3))            
        x_f = tf.reshape(X_c[:,0], shape=[-1,1])
        y_f = tf.reshape(X_c[:,1], shape=[-1,1])
        t_c = lhs(1, samples=self.collacation_points, criterion='m').astype(np.float32)
        t_c = tf.reshape(np.asarray(np.min(self.ParticleData.time_bound[0])+t_c*(self.ParticleData.time_bound[1]-self.ParticleData.time_bound[0])), shape=[-1,1])            
        vel = neural_net(t_c, x_f, y_f, vel_weights, vel_biases)

        u_x = tf.gradients(vel[:,0], x_f)
        u_y = tf.gradients(vel[:,0], y_f)
        u_t = tf.gradients(vel[:,0], t_c)

        v_x = tf.gradients(vel[:,1], x_f)
        v_y = tf.gradients(vel[:,1], y_f)
        v_t = tf.gradients(vel[:,1], t_c)

        p_x = tf.gradients(vel[:,2], x_f)
        p_y = tf.gradients(vel[:,2], y_f)

        u_xx = tf.gradients(u_x, x_f)
        u_yy = tf.gradients(u_y, y_f)

        v_xx = tf.gradients(v_x, x_f)
        v_yy = tf.gradients(v_y, y_f)

        ns_x = tf.reduce_sum(tf.square(tf.reshape(u_t,shape=[-1,1]) + \
            tf.reshape(vel[:,0], shape=[-1,1])*u_x + tf.reshape(vel[:,1], shape=[-1,1])*u_y + \
            [x/self.rho for x in p_x] - \
            [x/(self.rho*self.mu) for x in u_xx] - [x/(self.rho*self.mu) for x in u_yy]))

        ns_y = tf.reduce_sum(tf.square(tf.reshape(v_t,shape=[-1,1]) + \
            tf.reshape(vel[:,0], shape=[-1,1])*v_x + tf.reshape(vel[:,1], shape=[-1,1])*v_y + \
            [x/(self.rho) for x in p_y] - \
            [x/(self.rho*self.mu) for x in v_xx] - [x/(self.rho*self.mu) for x in v_yy]))

        cont = tf.reduce_sum(tf.square(u_x+v_y))

        return ns_x/self.collacation_points, ns_y/self.collacation_points, cont/self.collacation_points

    def vel_predict(self, t, x, y):
        scaled_pos = self.ParticleData.rescale_test(np.concatenate((x,y)))
        scaled_t = t/self.ParticleData.t_scale
        scaled_vel=neural_net(t,x,y,self.vel_weights, self.vel_biases)[:,:2]
        vel= scaled_vel*self.ParticleData.sigma_pos/self.ParticleData.max_time
        return self.sess.run(vel)
    
    def pos_predict(self, t1, t2, x, y):
        vel = self.vel_predict(t1, x, y)
        pos = vel*(t2-t1)
        return pos
