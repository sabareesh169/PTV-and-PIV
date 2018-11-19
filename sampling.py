def sampling_points(initial, final, pred, list_, sigma):
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
    for i in range(50):
        self.sess.run(self.optimizer_phy)
        self.sess.run(self.optimizer_vel, feed_dict={self.VelocityModel.vel_sample: true_vel_ch})
    print(self.sess.run(self.VelocityModel.loss_vel, feed_dict={self.VelocityModel.vel_sample: true_vel_ch}))


def sampling_theta(self, optimizer_vel, initial, final_index, t_initial, t_final):
    true_vel_ch = (final_index-initial)/(t_final-t_initial)
    optimize_theta(self, true_vel_ch)

def sampling_sigma(self, optimizer_sigma, pred, final_index):
    for i in range(50):
        self.sess.run(optimizer_sigma, feed_dict={self.VelocityModel.likelihood : np.sum((pred-final_index)**2)})
