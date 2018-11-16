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

def optimize_theta(optimzer_vel, optimizer_phy, true_vel_ch):
    for i in range(50):
        optimzer_vel.run(feed_dict={vel_sample: true_vel_ch})
        optimzer_phy.run()
        
def sampling_theta(optimzer_vel, optimzer_phy, initial, final_index, t_initial, t_final, c_l, c_t):
    sample_x_data_ch = final_index[:, 0][:, None].astype(np.float32)/c_l
    sample_y_data_ch = final_index[:, 1][:, None].astype(np.float32)/c_l
    
    true_vel_x_ch = (sample_x_data_ch - initial_x_data_ch)/(t_final_data_ch - t_initial_data_ch)
    true_vel_y_ch = (sample_y_data_ch - initial_y_data_ch)/(t_final_data_ch - t_initial_data_ch)
    true_vel_ch = np.concatenate([true_vel_x_ch, true_vel_y_ch], axis=1)
    optimize_theta(optimzer_vel, optimizer_phy, true_vel_ch)
