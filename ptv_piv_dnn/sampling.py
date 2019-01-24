"""
This is to run optimzation so as to sample points, theta and sigma alternatively (Gibb's sampling).

Author:
    Sabareesh Mamidipaka, Ilias Bilionis
Date:
    11/19/2018
"""


def sampling_points(initial, final, pred, list_, sigma):
    index = np.arange(initial.shape[0])
    for i in range(initial.shape[0]):
        points_index = np.array(list_[i])
        points = final[points_index]
        log_p = -np.sum((pred[i,:]-points)**2, axis=1)/(2*sigma**2) 
        cum_sum_q=np.exp(log_p-sc.misc.logsumexp(log_p)).cumsum()
        u=np.random.rand()
        idx=np.arange(len(cum_sum_q))
        index[i]=points_index[idx[u<cum_sum_q][0]]
    return index

def optimize_theta(self, true_vel_ch, n_iter):
    for i in range(n_iter):
        self.sess.run(self.optimizer_phy)
        self.sess.run(self.optimizer_vel, feed_dict={self.VelocityModel.vel_sample: true_vel_ch})
        self.sess.run(self.optimizer_vel, feed_dict={self.VelocityModel.vel_sample: true_vel_ch})

def sampling_theta(self, initial, final_index, t_initial, t_final, n_iter=50):
    true_vel_ch = (final_index-initial)/(t_final-t_initial)
    optimize_theta(self, true_vel_ch, n_iter)

def sampling_sigma(self, pred, final_index):
    for i in range(100):
        self.sess.run(self.optimizer_sigma, feed_dict={self.VelocityModel.likelihood : np.sum((pred-final_index)**2)})
    return np.sum((pred-final_index)**2), self.sess.run(self.VelocityModel.sigma)
