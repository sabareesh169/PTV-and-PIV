"""
Unit test demonstrating the use and application of the code.
We generate random data set of 100 points for a flow around a cylinder.
We take in a training data of 100 points and sample theta and sigma 500 times to get a trained velocity model.
We can then use this DNN to predict velocity at all points in space.
"""

## Loading training and testing data
train_data=pd.read_csv("foc_train_data.csv")
test_data=pd.read_csv("foc_test_data.csv")
train_data = train_data
train_data = train_data.values
test_data = test_data.values

initial = train_data[:,[0,1]].astype(np.float32)
final = train_data[:,[2,3]].astype(np.float32)
t_initial = train_data[0,6].astype(np.float32)
t_final = train_data[0,7]

test_initial = test_data[:,[0,1]].astype(np.float32)
test_final = test_data[:,[2,3]].astype(np.float32)

radius = .05
rho = 1000.
mu = 0.008
vel_layers = [3, 100, 100, 3]

##  Setting up ParticleData, VelocityModel and Training modules
data = ParticleData(initial, final, t_initial, t_final, radius)
test_velocity_model = VelocityModel(data, vel_layers, rho, mu)
test_training = TrainingProcess(data, test_velocity_model)

## Training for 500 iterations
test_training.train_velocity(n_iter=500)
pred = test_training.pos_predict(t_initial, t_final, initial[:,0][:,None], initial[:,1][:,None])
test_pred = test_training.pos_predict(t_initial, t_final, test_initial[:,0][:,None], test_initial[:,1][:,None])

## Final loss values
print('%.3e, %.3e, %.3e, %.3e, %.3e' %(test_training.sess.run(test_velocity_model.loss_vel, feed_dict={test_velocity_model.vel_sample: test_training.sess.run(test_velocity_model.vel_NN)}), \
    test_training.sess.run(test_velocity_model.loss_NS_x), test_training.sess.run(test_velocity_model.loss_NS_y), test_training.sess.run(test_velocity_model.loss_cont), k))

f, ax1 = plt.subplots(1, 1, figsize=(40,35))
ax1.set_xlim(0.25,0.6)
ax1.set_ylim(0,0.4)
for i in range(initial.shape[0]):
    ax1.arrow(initial[i,0], initial[i,1],(pred[i,0]-initial[i,0]), (pred[i,1]-initial[i,1]), width=0.0005, head_width=0.002)
    ax1.scatter(train_data[:,[0,2]][i,:], train_data[:,[1,3]][i,:])
plt.savefig('train_prediction.png')

f, ax1 = plt.subplots(1, 1, figsize=(40,35))
ax1.set_xlim(0.25,0.6)
ax1.set_ylim(0,0.4)
for i in range(test_initial.shape[0]):
    ax1.arrow(test_initial[i,0], test_initial[i,1],(test_pred[i,0]-test_initial[i,0]), (test_pred[i,1]-test_initial[i,1]), width=0.0005, head_width=0.002)
    ax1.scatter(test_data[:,[0,2]][i,:], test_data[:,[1,3]][i,:])
plt.savefig('test_prediction.png')

f, ax1 = plt.subplots(1, 1, figsize=(40,35))
ax1.set_xlim(0.25,0.6)
ax1.set_ylim(0,0.4)
for i in range(test_initial.shape[0]):
    ax1.arrow(test_initial[i,0], test_initial[i,1],(test_final[i,0]-test_initial[i,0]), (test_final[i,1]-test_initial[i,1]), width=0.0005, head_width=0.002)
plt.savefig('test_true.png')
