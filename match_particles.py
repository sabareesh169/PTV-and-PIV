"""
This scripts solves ...

Author:
    First, Last

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
    Initializes the NN.

    :param layers: What am I?
    :returns: A blah blah
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
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(200/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

## Forward propagation
def neural_net(t, x, y, weights, biases):
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
    
## Creating the neural network
vel_layers = [3, 100, 100, 100, 100, 100, 100, 3]
vel_weights, vel_biases = initialize_NN(vel_layers)
