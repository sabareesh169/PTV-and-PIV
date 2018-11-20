"""
This is to run optimzation so as to sample points, theta and sigma alternatively (Gibb's sampling).

Author:
    Sabareesh Mamidipaka, Ilias Bilionis
Date:
    11/19/2018
"""


def sampling_points(initial, final, pred, list_, sigma):
    """
    Sampling the points by assigning a pmf according to the current prediction of the DNN.

    :param array: the initial and final position, and the prediction and variance of the DNN
    :returns: matching index for each point in the initial data.
    """
        
    index = np.arange(initial.shape[0])
    for i in range(initial.shape[0]):
        points_index = np.array(list_[i])
        points = final[points_index]
        p = np.exp(-np.sum((pred[i,:]-points)**2, axis=1)/(2*sigma**2))/(2*sigma)
        cum_sum_q=np.exp(np.log(p)-sc.misc.logsumexp(np.log(p))).cumsum()
        u=np.random.rand()
        idx=np.arange(len(cum_sum_q))
        index[i]=points_index[idx[u<cum_sum_q][0]]
    return index

def optimize_theta(self, true_vel_ch):
    """
    Optimizing the DNN when the sample velocity is given.

    :param array: the current 'true velocity' given the matching index.
    """
    
    for i in range(50):
        self.sess.run(self.optimizer_phy)
        self.sess.run(self.optimizer_vel, feed_dict={self.VelocityModel.vel_sample: true_vel_ch})

def sampling_theta(self, initial, final_index, t_initial, t_final):
    """
    Optimizing the DNN when the sample velocity is given.

    :param array: the initial and final position, and the prediction and variance of the DNN
    :returns: matching index for each point in the initial data.
    """
    true_vel_ch = (final_index-initial)/(t_final-t_initial)
    optimize_theta(self, true_vel_ch)

def sampling_sigma(self, pred, final_index):
    """
    Optimizing the variance given the likelihood.

    :param array: the optimzer for variance, prediction and matched index given by the DNN
    """
    for i in range(50):
        self.sess.run(self.optimizer_sigma, feed_dict={self.VelocityModel.likelihood : np.sum((pred-final_index)**2)})
