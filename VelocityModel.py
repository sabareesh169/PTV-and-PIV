class VelocityModel:
    """
    Takes in the Particle data, charecteristics of the fluid and constructs a neural network with a given architecture
    to predict velocity.
    Calculates the residuals of the governing equations for the predicted velocity. 
    Also, calculates the probability of the variance given the likelihood value.
    """

    def __init__(self, ParticleData, vel_layers, rho, mu, collacation_points=1000):
        
        self.vel_weights, self.vel_biases = initialize_NN(vel_layers)
        self.rho = rho
        self.mu = mu

        self.vel_NN = neural_net(bs.t_initial_norm, bs.initial_pos[:,0][:,None], bs.initial_pos[:,1][:,None], self.vel_weights, self.vel_biases)

        self.vel_sample = tf.placeholder(tf.float32, shape=(bs.initial_pos.shape[0], 2))
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
        X_c = lhs(2, samples=collacation_points, criterion='m').astype(np.float32)
        X_c = (np.asarray((-1.5)+X_c*3))            
        x_f = tf.reshape(X_c[:,0], shape=[-1,1])
        y_f = tf.reshape(X_c[:,1], shape=[-1,1])
        t_c = lhs(1, samples=collacation_points, criterion='m').astype(np.float32)
        t_c = tf.reshape(np.asarray(np.min(bs.time_bound[0])+t_c*(bs.time_bound[1]-bs.time_bound[0])), shape=[-1,1])            
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
            [x/rho for x in p_x] - \
            [x/(rho*mu) for x in u_xx] - [x/(rho*mu) for x in u_yy]))

        ns_y = tf.reduce_sum(tf.square(tf.reshape(v_t,shape=[-1,1]) + \
            tf.reshape(vel[:,0], shape=[-1,1])*v_x + tf.reshape(vel[:,1], shape=[-1,1])*v_y + \
            [x/(rho) for x in p_y] - \
            [x/(rho*mu) for x in v_xx] - [x/(rho*mu) for x in v_yy]))

        cont = tf.reduce_sum(tf.square(u_x+v_y))

        return ns_x/collacation_points, ns_y/collacation_points, cont/collacation_points

