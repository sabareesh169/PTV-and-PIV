X_c = lhs(2, samples=100, criterion='m')
initial_pos = X_c*40.
final_pos = initial_pos+np.array([2.,0])
t_initial = np.zeros([initial_pos.shape[0], 1])
t_final = np.ones([initial_pos.shape[0], 1])*0.01
radius = 4.
rho = 1000.
mu = 0.008
vel_layers = [3, 100, 100, 2]

X_c = lhs(2, samples=100, criterion='m')
test_initial_pos = X_c*40.
test_final_pos = test_initial_pos+np.array([2.,0])
test_true_vel = (test_final_pos - test_initial_pos)/(t_final - t_initial)

test_data = ParticleData(initial_pos, final_pos, t_initial, t_final, radius)
test_velocity_model = VelocityModel(test_data, vel_layers, rho, mu)
test_training = TrainingProcess(test+data, test_velocity_model)

with tf.Session() as sess:
  test_training.predict_velocity(n_iter=500)
  test_vel_pred = test_training.vel_predict(t_initial, t_final, test_initial_pos, test_final_pos)
  print(mean_squared_error(test_vel_pred, test_true_vel))
  
