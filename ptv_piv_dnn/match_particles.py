"""
Construction initialization and forward propogation of the NN.

Author:
    Sabareesh Mamidipaka, Ilias Bilionis
Date:
    11/19/2018
"""


import tensorflow as tf


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
    xavier_stddev = np.sqrt(2./(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

def neural_net(t, x, y, weights, biases):
    """
    Forward propogation of the Neural Network
    :param : Time and position of the particle and the parameters of the network
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
