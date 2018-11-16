from pyDOE import lhs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import pandas as pd
import scipy.io


class PIV:

    def __init__(self, initial, final, t_initial, t_final, vel_layers, radius):
        self.sess = tf.Session()
        self.initial = initial.astype(np.float32)
        self.final = final.astype(np.float32)
        self.t_initial = t_initial.astype(np.float32)
        self.t_final = t_final.astype(np.float32)

        self.vel_weights, self.vel_biases = initialize_NN(vel_layers)

        self.initial_ch, self.mu_train_pos, self.sigma_train_pos=rescale(initial)
        self.final_ch=rescale_test(final, self.mu_train_pos, self.sigma_train_pos)
        self.t_scale = np.max(t_final)
        self.t_initial_ch =t_initial/self.t_scale
        self.t_final_ch = t_final/self.t_scale

        tree=spatial.KDTree(initial)
        self.list_=tree.query_ball_point(initial, radius)
        self.vel_pred = neural_net(self.t_initial_ch, self.initial_ch[:,0][:,None], self.initial_ch[:,1][:,None], self.vel_weights, self.vel_biases)

        self.vel_sample = tf.placeholder(tf.float32, shape=(initial.shape[0], 2))
        self.loss_vel = tf.reduce_sum(tf.square(self.vel_pred[:,:2] - self.vel_sample))
        self.loss_NS_x, self.loss_NS_y, self.loss_cont = residue(self.vel_weights, self.vel_biases, np.min(self.t_initial_ch), self.t_scale)
        self.phy_loss = self.loss_NS_x + self.loss_NS_y + self.loss_cont
        alpha = tf.constant(0.001, dtype=tf.float32)
        beta = tf.constant(1., dtype=tf.float32)
        self.sigma = tf.Variable(1., dtype=tf.float32)
        self.likelihood = tf.placeholder(dtype=tf.float32)
        self.neg_log_prob = (2*alpha+initial.shape[0]+2)*tf.log(self.sigma)+(beta+self.likelihood/2)/self.sigma**2
        self.optimizer_phy = tf.train.AdamOptimizer().minimize(self.phy_loss, var_list=self.vel_weights+self.vel_biases)
        self.optimizer_vel = tf.train.AdamOptimizer().minimize(self.loss_vel, var_list=self.vel_weights+self.vel_biases)
        self.optimizer_sigma = tf.train.AdamOptimizer().minimize(self.neg_log_prob, var_list=self.sigma)
        self.pos_pred=self.vel_pred[:,:2]*(t_final- t_initial)*self.sigma_train_pos/np.max(t_final)+initial
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def PTV(self, n_iter):
        for i in range(n_iter):
            print(i)
            index=sampling_points(initial, final, self.sess.run(self.pos_pred), self.list_, self.sess.run(self.sigma))
            final_index=rescale_test(self.final[index], self.mu_train_pos, self.sigma_train_pos)
            sampling_theta(self, self.optimizer_vel, self.initial, final_index, self.t_initial, self.t_final)
            sampling_sigma(self, self.optimizer_sigma, self.sess.run(self.pos_pred), final_index)
        self.index=index            

    def PIV(self, n_iter):
        for i in range(n_iter):
            sampling_theta(self.optimzer_vel, self.optimzer_phy, self.initial, self.final, self.t_initial, self.t_final)

    def vel_predict(self, t, x, y):
        scaled_pos= rescale_test(np.concatenate((x,y)), self.mu_train_pos, self.sigma_train_pos)
        scaled_t= t/self.t_scale
        scaled_vel=neural_net(t,x,y,self.vel_weights, self.vel_biases)[:,:2]
        vel= scaled_vel*self.sigma_train_pos/self.t_scale
        return self.sess.run(vel)
    
    def pos_predict(self, t1, t2, x, y):
        vel = self.vel_predict(t1, x, y)
        pos = vel*(t2-t1)
        return pos
    

