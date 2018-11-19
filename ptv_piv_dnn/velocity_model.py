"""
Construction and setting up of loss functions of DNN to predict the correct velocity.
Author:
    Sabareesh Mamidipaka, Ilias Bilionis
Date:
    11/19/2018
"""


__all__ = ['VelocityModel']

import tensorflow as tf
import numpy as np
from pyDOE import lhs


class VelocityModel:
    """
    Constructs a neural network for the prediction of velocity.
    Calculates the residuals for the governing equations at random collacation points.
    Assign a probability function for the matched particles.
    Assign a probability function for the value of sigma.
    """

    def __init__(self, ParticleData, vel_layers, rho, mu, collacation_points=1000):
        
        self.vel_weights, self.vel_biases = initialize_NN(vel_layers)
        self.rho = rho
        self.mu = mu
        self.collacation_points=collacation_points
        self.ParticleData=ParticleData
        
        self.vel_NN = neural_net(ParticleData.t_initial_norm, ParticleData.initial_pos_norm[:,0][:,None], ParticleData.initial_pos_norm[:,1][:,None], self.vel_weights, self.vel_biases)[:,:2]
        self.pos_NN = ParticleData.initial_pos + self.vel_NN*self.ParticleData.sigma_pos*(ParticleData.t_initial- ParticleData.t_final)/self.ParticleData.max_time

        self.vel_sample = tf.placeholder(tf.float32, shape=(ParticleData.initial_pos.shape[0], 2))
        self.loss_vel = tf.reduce_sum(tf.square(self.vel_NN - self.vel_sample))
        self.loss_NS_x, self.loss_NS_y, self.loss_cont = self.residue(self.vel_weights, self.vel_biases)
        self.total_residue = self.loss_NS_x + self.loss_NS_y + self.loss_cont
        alpha = tf.constant(0.001, dtype=tf.float32)
        beta = tf.constant(1., dtype=tf.float32)
        self.sigma = tf.Variable(1., dtype=tf.float32)
        self.likelihood = tf.placeholder(dtype=tf.float32)
        self.neg_log_prob = (2*alpha+102)*tf.log(self.sigma)+(beta+self.likelihood/2)/self.sigma**2
        
    def residue(self, vel_weights, vel_biases):
        """
        This is to calculate the reiduals of the governing equations at random collacation points 
        
        :param size: parameters of the network.
        :returns: Residue of the governing equations for the given network.
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
        """
        This is to predict the velocity at given position and time
        
        :param : time and position 
        :returns: predicted velocity of the fluid at that position and time
        """
        scaled_pos = self.ParticleData.scale_pos_data(np.concatenate((x,y),axis=1))
        scaled_t = self.ParticleData.scale_time_data(t,x)
        scaled_vel=neural_net(scaled_t, scaled_pos[:,0][:,None], scaled_pos[:,1][:,None], self.vel_weights, self.vel_biases)[:,:2]
        vel= scaled_vel*self.ParticleData.sigma_pos/self.ParticleData.max_time
        return vel
    
    def pos_predict(self, t1, t2, x, y):
        """
        This is to predict the particle position at final time given the initial position and time of particle.
        
        :param : the initia time and position of the particle
        :returns: the predicted final position of the particle
        """
        vel = self.vel_predict(t1, x, y)
        pos = np.concatenate((x,y),axis=1) + vel*(t2-t1)
        return pos
