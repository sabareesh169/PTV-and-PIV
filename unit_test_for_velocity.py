"""
Unit test demonstrating the use and application of the code.
We generate random data set of 100 points for a uniform flow field in x-direction(200 m/s).
We take in a training data of 100 points and sample theta and sigma 500 times to get a trained velocity model.
We can then use this DNN to predict velocity at all points in space.
"""

##  Generating a training data of uniformly spread particles in a 40*40 space
X_c = lhs(2, samples=100, criterion='m')
initial_pos = X_c*40.
final_pos = initial_pos+np.array([2.,0])
t_initial = 0.
t_final = 0.01
radius = 4.
rho = 1000.
mu = 0.008
vel_layers = [3, 100, 100, 3]

##  Setting up ParticleData, VelocityModel and Training modules
test_data = ParticleData(initial_pos, final_pos, t_initial, t_final, radius)
test_velocity_model = VelocityModel(test_data, vel_layers, rho, mu)
test_training = TrainingProcess(test_data, test_velocity_model)

##  Test data set
x_plot = np.linspace(0, 40, 100)
y_plot = np.linspace(0, 40, 100)
X,Y = np.meshgrid(x_plot, y_plot)

## Training for 500 iterations
test_training.train_velocity(n_iter=500)
test_vel_pred = test_training.vel_predict(t_initial, X.reshape(-1,1), Y.reshape(-1,1))
test_abs_vel_pred = np.linalg.norm(test_vel_pred, axis=1)

fig, ax = plt.subplots()

p = ax.pcolor(X, Y, test_abs_vel_pred.reshape(x_plot.shape[0],-1), vmin=abs(test_abs_vel_pred).min(), vmax=abs(test_abs_vel_pred).max())
cb = fig.colorbar(p)
plt.savefig('unit_test.png')
