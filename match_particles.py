"""
This scripts does both PTV and PIV given the locations of points. It gives out a continuous function for velocity and pressure. 

Author:
    Ilias Bilionis, Sabareesh Mamidipaka

Date:
    11/15/2018
"""


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
    
## Defining the structure of neural network and intitializing network parameters.
vel_layers = [3, 100, 100, 100, 100, 100, 100, 3]
vel_weights, vel_biases = initialize_NN(vel_layers)

## Charecteristic scales
c_l=0.05
c_t=0.001
c_v=c_l/c_t

## Load data
train_data=pd.read_csv("foc_train_data.csv")
test_data=pd.read_csv("foc_val_data.csv")
train_data = train_data
train_data = train_data.values
test_data = test_data.values

## Training data
initial = train_data[:,[0,1]].astype(np.float32)
initial_x_data = tf.constant(initial[:, 0][:, None], name='x_init', dtype=tf.float32)
initial_y_data = tf.constant(initial[:, 1][:, None], name='y_init', dtype=tf.float32)

final = train_data[:,[2,3]].astype(np.float32)
final_x_data = tf.constant(final[:, 0][:, None], name='x_final', dtype=tf.float32)
final_y_data = tf.constant(final[:, 1][:, None], name='y_final', dtype=tf.float32)

t_initial_data = tf.constant(train_data[:,6].astype(np.float32)[:, None], name='t_init')
t_final_data = tf.constant(train_data[:,7].astype(np.float32)[:, None], name='t_final')

## Training data in charescteristic length scales
initial_x_data_ch = tf.constant(initial[:, 0][:, None], name='x_init_ch', dtype=tf.float32)/c_l
initial_y_data_ch = tf.constant(initial[:, 1][:, None], name='y_init_ch', dtype=tf.float32)/c_l
initial_ch = tf.concat([initial_x_data, initial_y_data], 1)/c_l 

final_x_data_ch = tf.constant(final[:, 0][:, None], name='x_final_ch', dtype=tf.float32)/c_l
final_y_data_ch = tf.constant(final[:, 1][:, None], name='y_final_ch', dtype=tf.float32)/c_l
final_ch = tf.concat([final_x_data, final_y_data], 1)/c_l 

t_initial_data_ch = tf.constant(train_data[:,6].astype(np.float32)[:, None], name='t_init_ch')/c_t
t_final_data_ch = tf.constant(train_data[:,7].astype(np.float32)[:, None], name='t_final_ch')/c_t

## Testing data
test_initial = test_data[:,[0,1]].astype(np.float32)
test_initial_x_data = tf.constant(test_initial[:, 0][:, None], name='test_x_init', dtype=tf.float32)
test_initial_y_data = tf.constant(test_initial[:, 1][:, None], name='test_y_init', dtype=tf.float32)

test_final = test_data[:,[2,3]].astype(np.float32)
test_final_x_data = tf.constant(test_final[:, 0][:, None], name='x_final', dtype=tf.float32)
test_final_y_data = tf.constant(test_final[:, 1][:, None], name='y_final', dtype=tf.float32)

test_t_initial_data = tf.constant(test_data[:,6].astype(np.float32)[:, None], name='t_init')
test_t_final_data = tf.constant(test_data[:,7].astype(np.float32)[:, None], name='t_final')

## Testing data in charescteristic length scales
test_initial_x_data_ch = tf.constant(test_initial[:, 0][:, None], name='test_x_init_ch', dtype=tf.float32)/c_l
test_initial_y_data_ch = tf.constant(test_initial[:, 1][:, None], name='test_y_init_ch', dtype=tf.float32)/c_l
test_initial_ch = tf.concat([test_initial_x_data_ch, test_initial_y_data_ch], 1)

test_final_x_data_ch = tf.constant(test_final[:, 0][:, None], name='x_final_ch', dtype=tf.float32)/c_l
test_final_y_data_ch = tf.constant(test_final[:, 1][:, None], name='y_final_ch', dtype=tf.float32)/c_l
test_final_ch = tf.concat([test_final_x_data_ch, test_final_y_data_ch], 1)

test_t_initial_data_ch = tf.constant(test_data[:,6].astype(np.float32)[:, None], name='t_init_ch')/c_t
test_t_final_data_ch = tf.constant(test_data[:,7].astype(np.float32)[:, None], name='t_final_ch')/c_t


## Selecting the final points within a radius for each initial particle.
tree=spatial.KDTree(initial)
list_=tree.query_ball_point(initial, 0.1)

## Velocity for the train and test particles
vel_pred = neural_net(t_initial_data_ch, initial_x_data_ch, initial_y_data_ch, vel_weights, vel_biases)
test_vel_pred = neural_net(test_t_initial_data_ch, test_initial_x_data_ch, test_initial_y_data_ch, vel_weights, vel_biases)

with tf.Session() as sess:
    true_vel_x_ch = sess.run((final_x_data_ch - initial_x_data_ch)/(t_final_data_ch - t_initial_data_ch))
    true_vel_y_ch = sess.run((final_y_data_ch - initial_y_data_ch)/(t_final_data_ch - t_initial_data_ch))
    true_vel_ch = np.concatenate([true_vel_x_ch, true_vel_y_ch], axis=1)
    vel_placeholder = tf.placeholder(tf.float32, shape=(initial.shape[0], 2))
    loss_vel = tf.reduce_sum(tf.square(vel_pred[:,:2] - vel_placeholder))
    loss_NS_x, loss_NS_y, loss_cont = NS(vel_weights, vel_biases, lb, ub)
    phy_loss = loss_NS_x + loss_NS_y + loss_cont
    alpha = tf.constant(0.001, dtype=tf.float32)
    beta = tf.constant(1., dtype=tf.float32)
    sigma = tf.Variable(1., dtype=tf.float32)
    likelihood = tf.placeholder(dtype=tf.float32)
    neg_log_prob = (2*alpha+102)*tf.log(sigma)+(beta+likelihood/2)/sigma**2
    optimizer_ph = tf.train.AdamOptimizer().minimize(phy_loss, global_step=global_step, var_list=vel_weights+vel_biases)
    optimizer_vel = tf.train.AdamOptimizer().minimize(loss_vel, global_step=global_step, var_list=vel_weights+vel_biases)
    optimzer_sigma = tf.train.AdamOptimizer().minimize(neg_log_prob, global_step=global_step, var_list=sigma)
    init = tf.global_variables_initializer()
    sess.run(init)
    k=0
