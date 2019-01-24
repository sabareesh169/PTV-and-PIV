"""
Optimization of DNN to predict the correct velocity.

Author:
    Sabareesh Mamidipaka, Ilias Bilionis
    
Date:
    11/19/2018
"""

__all__ = ['TraningProcess']


import tensorflow as tf
import numpy as np


class TrainingProcess:
    """
    
    Takes in the loss functions constructed in the Velocity Model.
    Runs an optimizer to minimize the loss function
    """
    
    def __init__(self, ParticleData, VelocityModel):
        self.sess = tf.Session()

        self.ParticleData = ParticleData
        self.VelocityModel = VelocityModel
        self.optimizer_phy = tf.train.AdamOptimizer().minimize(self.VelocityModel.total_residue, var_list=self.VelocityModel.vel_weights+self.VelocityModel.vel_biases)
        self.optimizer_vel = tf.train.AdamOptimizer().minimize(self.VelocityModel.loss_vel, var_list=self.VelocityModel.vel_weights+self.VelocityModel.vel_biases)
        self.optimizer_sigma = tf.train.AdamOptimizer().minimize(self.VelocityModel.neg_log_prob, var_list=self.VelocityModel.dummy_var)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def match_particles_and_velocity(self, n_iter):
        """
        This is to optimze in the case when the matching points are not known.
        We sample points, optmize theta, and sigma alternatively one after the other.
        
        :param size: Sampling the points, theta and sigma one after the other for n_iter number of times.
        :returns: optimized weights of the velocity DNN and the matched particles.
        """

        for i in range(n_iter):
            index = sampling_points(self.ParticleData.initial_pos, self.ParticleData.final_pos, self.sess.run(self.VelocityModel.pos_NN), self.ParticleData.cluster, self.sess.run(self.VelocityModel.sigma))
            final_index = self.ParticleData.scale_pos_data(self.ParticleData.final_pos[index])
            matched_points = 1-float(np.count_nonzero(np.arange(initial.shape[0])-index)) / initial.shape[0]
            sampling_theta(self, self.ParticleData.initial_pos_norm, final_index, self.ParticleData.t_initial_norm, self.ParticleData.t_final_norm)
            like, sigma =sampling_sigma(self, self.sess.run(self.VelocityModel.pos_NN), self.ParticleData.final_pos[index])
#            print('iteration %i number of matched points:' %i, matched_points*100, 'likelihood:', like, 'sigma:', sigma)
        return index, sigma

    def train_velocity(self, true_vel, n_iter):
        """
        This is to optimze in the case when the matching points are known.
        We optmize theta keeping sigma fixed at the initial value.
        
        :param size: Sampling the theta and sigma one after the other for n_iter number of times.
        :returns: optimized weights of the velocity DNN.
        """

        true_vel_ch = true_vel * self.ParticleData.max_time/self.ParticleData.sigma_pos        
        for i in range(n_iter*50):
            self.sess.run(self.optimizer_vel, feed_dict={self.VelocityModel.vel_sample: true_vel_ch})
#            sampling_theta(self, self.ParticleData.initial_pos_norm, self.ParticleData.final_pos_norm, self.ParticleData.t_initial_norm, self.ParticleData.t_final_norm)

    def match_points(self, true_vel, n_iter):
        """
        This is to match the index the case when the true velocity is known.
        We optmize theta, and sigma alternatively one after the other.
        
        :param size: Sampling the theta and sigma one after the other for n_iter number of times.
        :returns: optimized weights of the velocity DNN.
        """
        
        true_vel_ch = true_vel * self.ParticleData.max_time/self.ParticleData.sigma_pos
        for i in range(n_iter):
            optimize_theta(self, true_vel_ch)
        return sampling_points(self.ParticleData.initial_pos, self.ParticleData.final_pos, self.sess.run(self.VelocityModel.pos_NN), self.ParticleData.cluster, self.ParticleData.radius/15.), self.sess.run(self.VelocityModel.sigma)       

    def vel_predict(self, t, pos):
        """
        This is to predict the velocity at given time and position.
        
        :param : the time and position of the point.
        :returns: velocity at the particular position and time.
        """
        
        return self.sess.run(self.VelocityModel.vel_predict(t,pos))                   

    def pos_predict(self, t1, t2, pos):
        """
        This is to predict the particle position at final time given the initial position and time of particle.
        :param : the time and position of the point and the time at whoch the position is required.
        :returns: predicted position of the particle.
        """
        return self.sess.run(self.VelocityModel.pos_predict(t1, t2, pos))  
