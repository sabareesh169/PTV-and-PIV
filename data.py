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
        
        self.c_l = np.float32(np.max(ub[1:]))
        self.c_t = np.float32(np.max(ub[0]))

        self.initial_x_data = initial[:, 0][:, None].astype(np.float32)
        self.initial_y_data = initial[:, 1][:, None].astype(np.float32)
 
        self.initial_x_data_ch = tf.constant(initial[:, 0][:, None], name='x_init_ch', dtype=tf.float32)/c_l
        self.initial_y_data_ch = tf.constant(initial[:, 1][:, None], name='y_init_ch', dtype=tf.float32)/c_l
        self.initial_ch = tf.concat([self.initial_x_data, self.initial_y_data], 1)/c_l 

        self.final_x_data = tf.constant(final[:, 0][:, None], name='x_final', dtype=tf.float32)
        self.final_y_data = tf.constant(final[:, 1][:, None], name='y_final', dtype=tf.float32)

        self.final_x_data_ch = tf.constant(final[:, 0][:, None], name='x_final_ch', dtype=tf.float32)/c_l
        self.final_y_data_ch = tf.constant(final[:, 1][:, None], name='y_final_ch', dtype=tf.float32)/c_l
        self.final_ch = tf.concat([self.final_x_data, self.final_y_data], 1)/c_l 

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



