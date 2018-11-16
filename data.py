from pyDOE import lhs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import pandas as pd
import scipy.io


def initialize_NN(layers):
    """
    Structure and initilization of the NN.
    :param layers: A list of neurons in each layer(including the input and output layers)
    :returns: The intialized weights and biases of the required choice of the hidden layers
    """
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W = xavier_initialisation(size=[layers[l], layers[l+1]])
        b = tf.Variable(tf.zeros([1,layers[l+1]]), dtype=tf.float32)
        weights.append(W)
        biases.append(b)        
    return weights, biases
    
def xavier_initialisation(size):
    """
    Initializes the NN.
    :param size: The dimensions
    :returns: The intialized weights or biases for one specific hidden layer
    """
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(200/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

def neural_net(t, x, y, weights, biases):
    """
    Forward propogation of the Neural Network
    :param size: Time and position of the particle and the parameters of the network
    :returns: Velocity and pressure
    """
    X = tf.concat([t,x,y],1)
    num_layers = len(weights) 
    H = X
    for l in range(num_layers):
        W = weights[l]
        b = biases[l]
        H = tf.add(tf.matmul(H, W), b)
        if l == num_layers - 1:
            H = tf.identity(H, name='velocity')
        else:
            H = tf.tanh(H)
    return H

## Residual of NS and contiuity equations
def residue(vel_weights, vel_biases, lb, ub):
    """
    :param size: parameters of the network, the bounds on space and time for the data.
    :returns: Residue of the governing equations.
    """
    X_c = lhs(2, samples=6000, criterion='m')
    X_c = (np.asarray(lb[1:])+X_c*(np.asarray(ub[1:])-np.asarray(lb[1:]))).astype(np.float32)            
    x_f = tf.reshape(X_c[:,0], shape=[-1,1])
    y_f = tf.reshape(X_c[:,1], shape=[-1,1])
    t_f = tf.ones(dtype=tf.float32, shape=[6000,1])*np.asarray(lb[0])/c_t
    vel = neural_net(t_f, x_f, y_f, vel_weights, vel_biases)

    u_x = tf.gradients(vel[:,0], x_f)
    u_y = tf.gradients(vel[:,0], y_f)
    u_t = tf.gradients(vel[:,0], t_f)

    v_x = tf.gradients(vel[:,1], x_f)
    v_y = tf.gradients(vel[:,1], y_f)
    v_t = tf.gradients(vel[:,1], t_f)

    p_x = tf.gradients(vel[:,2], x_f)
    p_y = tf.gradients(vel[:,2], y_f)

    u_xx = tf.gradients(u_x, x_f)
    u_yy = tf.gradients(u_y, y_f)

    v_xx = tf.gradients(v_x, x_f)
    v_yy = tf.gradients(v_y, y_f)

    ns_x = tf.reduce_sum(tf.square(tf.reshape(u_t,shape=[-1,1]) + \
        tf.reshape(vel[:,0], shape=[-1,1])*u_x + tf.reshape(vel[:,1], shape=[-1,1])*u_y + \
        [x/(1000*c_v**2) for x in p_x] - \
        [x*0.8*10**(-6)/(c_v*c_l) for x in u_xx] - [x*0.8*10**(-6)/(c_v*c_l) for x in u_yy]))

    ns_y = tf.reduce_sum(tf.square(tf.reshape(v_t,shape=[-1,1]) + \
        tf.reshape(vel[:,0], shape=[-1,1])*v_x + tf.reshape(vel[:,1], shape=[-1,1])*v_y + \
        [x/(1000*c_v**2) for x in p_y] - \
        [x*0.8*10**(-6)/(c_v*c_l) for x in v_xx] - [x*0.8*10**(-6)/(c_v*c_l) for x in v_yy]))

    cont = tf.reduce_sum(tf.square(u_x+v_y))

    return ns_x/6000., ns_y/6000., cont/6000.

def sampling_points(initial, final, pred, list_, sigma):
    index = np.arange(initial.shape[0])
    for i in range(initial.shape[0]):
        points_index = np.array(list_[i])
        points = final[points_index]
        p = np.exp(-np.sum((pred[i,:]-points)**2, axis=1)/(2*sigma_**2))/(2*sigma_)
        cum_sum_q=np.exp(np.log(p)-sc.misc.logsumexp(np.log(p))).cumsum()
        u=np.random.rand()
        idx=np.arange(len(cum_sum_q))
        index[i]=points_index[idx[u<cum_sum_q][0]]
    return index

def sampling_sigma(optimizer_sigma, pred, final_index):
    for i in range(50):
        optimzer_sigma.run(feed_dict={likelihood: np.sum((pred-final[index])**2)})

def sampling_theta(optimzer_theta, initial, final_index, t_initial, t_final, c_l, c_t):
            sample_x_data_ch = tf.constant(final[index][:, 0][:, None], dtype=tf.float32)/c_l
        sample_y_data_ch = tf.constant(final[index][:, 1][:, None], dtype=tf.float32)/c_l
        final_ch = tf.concat([sample_x_data_ch, sample_y_data_ch], 1)/c_l 

        true_vel_x_ch = sess.run((sample_x_data_ch - initial_x_data_ch)/(t_final_data_ch - t_initial_data_ch))
        true_vel_y_ch = sess.run((sample_y_data_ch - initial_y_data_ch)/(t_final_data_ch - t_initial_data_ch))
        true_vel_ch = np.concatenate([true_vel_x_ch, true_vel_y_ch], axis=1)

class PIV:

    def __init__(self, initial, final, t_initial, t_final, vel_layers, lb, ub, c_l, c_t, radius):
        self.sess = tf.Session()
        self.initial = initial
        self.final = final
        self.t_initial = t_initial
        self.t_final = t_final
        self.lb = lb
        self.ub = ub
        self.vel_layers = vel_layers
        self.vel_weights, self.vel_biases = initialize_NN(vel_layers)
        self.c_l = c_l
        self.c_t = c_t

        self.initial_x_data = tf.constant(initial[:, 0][:, None], name='x_init', dtype=tf.float32)
        self.initial_y_data = tf.constant(initial[:, 1][:, None], name='y_init', dtype=tf.float32)
 
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



