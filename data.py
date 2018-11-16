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

    def __init__(self, initial, final, t_initial, t_final, vel_layers, lb, ub, radius):
        self.sess = tf.Session()
        self.initial = initial
        self.final = final
        self.t_initial = t_initial
        self.t_final = t_final
        self.lb = lb
        self.ub = ub
        self.vel_layers = vel_layers
        self.vel_weights, self.vel_biases = initialize_NN(vel_layers)

        self.initial = train_data[:,[0,1]].astype(np.float32)
 
        self.initial_ch, self.mu_train_pos, self.sigma_train_pos=rescale(initial)

        self.final = train_data[:,[2,3]].astype(np.float32)

        self.final_ch=rescale(final, self.mu_train_pos, self.sigma_train_pos)

        self.t_initial_ch, self.mu_train_t, self.sigma_train_t=rescale(t_initial)
        self.t_final_ch = rescale_test(t_final, self.mu_train_t, self.sigma_train_t)

        self.t_initial_data = tf.constant(t_initial.astype(np.float32)[:, None], name='t_init')
        self.t_final_data = tf.constant(t_final.astype(np.float32)[:, None], name='t_final')

        self.t_initial_data_ch = tf.constant(self.t_initial_data)/c_t
        self.t_final_data_ch = tf.constant(self.t_final_data)/c_t

        tree=spatial.KDTree(initial)
        self.list_=tree.query_ball_point(initial, radius)
        self.vel_pred = neural_net(self.t_initial_data_ch, self.initial_x_data_ch, self.initial_y_data_ch, self.vel_weights, self.vel_biases)

        ph = tf.placeholder(tf.float32, shape=(initial.shape[0], 2))
        self.loss_vel = tf.reduce_sum(tf.square(vel_pred[:,:2] - ph))
        self.loss_NS_x, self.loss_NS_y, self.loss_cont = NS(vel_weights, vel_biases, lb, ub)
        self.phy_loss = self.loss_NS_x + self.loss_NS_y + self.loss_cont
        alpha = tf.constant(0.001, dtype=tf.float32)
        beta = tf.constant(1., dtype=tf.float32)
        self.sigma = tf.Variable(1., dtype=tf.float32)
        likelihood = tf.placeholder(dtype=tf.float32)
        self.neg_log_prob = (2*alpha+102)*tf.log(self.sigma)+(beta+likelihood/2)/self.sigma**2
        self.optimizer_ph = tf.train.AdamOptimizer().minimize(phy_loss, global_step=global_step, var_list=vel_weights+vel_biases)
        self.optimizer_vel = tf.train.AdamOptimizer().minimize(loss_vel, global_step=global_step, var_list=vel_weights+vel_biases)
        self.optimzer_sigma = tf.train.AdamOptimizer().minimize(neg_log_prob, global_step=global_step, var_list=sigma)
        init = tf.global_variables_initializer()
        sess.run(init)



