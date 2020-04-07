# Physics informed Deep Learning to make velocity predictions

## Introduction
The purpose of this project is to find the fluid velocity from the images of a particle laden flow. There is a lot of research done into this problem which is called the Particle Image Velocimetry(PIV)/Particle Tracking Velocimetry(PTV) in Mechanical Engineering. The general methods followed for PIV is to break the field in frame 1 into interrogation windows and find the window with the highest correlation in the second frame to approximate the velocity at the position of interrogation window.

![General methodology followed for PIV](https://www.researchgate.net/profile/Daniel_Cote2/publication/50268148/figure/fig2/AS:305818365906946@1449924188734/The-underlying-concept-of-the-two-frames-cross-correlation-particle-image-velocimetry.png)

General methodology followed for PIV

As we can guess, the window size is very critical for the measurement. Increasing the size of the window increases the accuracy of measurements but reduces the resolution as we wont get any finer details of the fluid flow within the window. Particle Tracking Velocimetry uses fewer particles to be able to increase the resolution of the predictions but becomes more and more difficult to track particles and the predictions are far and few in between. So, there is a tradeoff between accuracy and resolution in both cases. 

The objective of this project is to use AI to solve this tradeoff.

## Approach 1
The idea is to apply deep learning to make predictions. Given the velocity measurements from PIV, we can fit a Deep Neural Network on this data to get an enhanced resolution data of the fluid velocity. But the drawback of this approach (which is essentially curve fitting) is that it tends to break some essential conditions like boundary conditions, initial conditions or the governing equations of the flow. 

![The inputs of the network time and position and the network spits out velocity in x and y directions](https://user-images.githubusercontent.com/25951391/78628223-67a3fc00-7848-11ea-9470-d5606d15df9b.png)

The inputs of the network time and position and the network spits out velocity in x and y directions

So, we need to implement the governing equations of the flow on the neural network as well. A Deep Neural Network performs backpropogation to change the weights of the network in order to minimize a prediction loss. In a normal regression modeling approach, the loss would be rmse between the actual values and the predictions (in our case between measurements from PIV/PTV and predictions of DNN). We now add the residual of the governing equations of the fluid to the loss function as well. For example, one of the governing equations we may want to implement is the 'conservation of mass' or 'continuity equation'. 
![The continutity equation](https://user-images.githubusercontent.com/25951391/78628442-f9ac0480-7848-11ea-95eb-0543ecddb8ff.png)

The continutity equation

![The residual of the continuity equation](https://user-images.githubusercontent.com/25951391/78628443-fa449b00-7848-11ea-9918-e811f235eb63.png)

The residual of the continuity equation

By adding the residual to the loss function, we minimize the residual which is effectively implementing the continuity equation on the predictions. The drawback of this method is that we are building on the measurements of PIV/PTV which by itself is not accurate and prne to errors. So, we need to come up with a method to match the particles using ML or any other method.

## Approach 2
By not relying on PIV/ PTV at all, the number of variables increase to matching positions vector, uncertainity in the position or search radius for each particle and the weights of the network. It's not easy to come up with one loss function to arrive at the optimal point. So, we use the technique of Gibbs sampling. Gibbs sampling essentially states that we can arrive at the optimal pint by sampling over each conditional distribution instead of the whole multivariate distribution.

Accordingly, for each particle in frame 1, we assign a particle in frame 2 wihtin a search radius using a Gaussian distribution around the predicted position of the neural network. Then we perform Gibbs sampling. We optimize the weights of the network according to the matched positions. We then update the search radius to maximize the likelihood and draw a sample of the matching particles again. We repeat this process continuously to finally settle on the optimal solution.

![image](https://user-images.githubusercontent.com/25951391/78709324-500c5800-78c8-11ea-9d66-e1f5fc048987.png)

Algorithm flowchart.

## Code 
The entire module is broken down into four parts. 
- The 'particle_data' file contains the ParticleData class. This class reads the data regarding the particle positions and derives the necessary information including the minimium, maximium, bounds of space and time, etc. Additonally also has methods to scale the data. - - The 'velocity_model' file contains the VelocityData class which takes in the particledata object, initializes a Neural Network, calculates the residue of the governing equations and also predicts the velocity and positon of particles. 
- The 'match_particles' file contains the functions necessary for initalizing and feed forwarding the neural networks. 
- The file 'sampling.py' contains all the functions necessary for sampling the matching points, the weights of the neural networkand to update the search radius of the particles.
- The file 'training_process.py' contains the TrainingProcess class which is responsible for the training steps depending on the context and also returns the final predicted position and velocity.

## Results

![image](https://user-images.githubusercontent.com/25951391/78709634-d2951780-78c8-11ea-92d8-65fedab14a46.png)

Consider the simulation shown above.

![image](https://user-images.githubusercontent.com/25951391/78709402-73370780-78c8-11ea-9841-6c45f39c201b.png)

The above picture contains the change in position of the particles. The model recieves the two end points of each vector representing the inital and final positions without any order. Point to note is that the positions of the particle alone do not give away the position of the cylinder in the field.

Below are the results obtained from the particle data above:
![image](https://user-images.githubusercontent.com/25951391/78709962-61a22f80-78c9-11ea-8694-6cd06b703618.png)

The top figure contains the predictions and the bottom figure contains the actual results. We can see that the algorithm actually infers the cylinder based on the particle positions and calculates the flow accordingly!

## Drawback
The above algorithm methodology work exceedingly well for the simulated data shown. But in case of real world applications, we cannot get the data with the same accuracy. In particular, we have the problem of missing particles (some of them cannot be detected) in either of the frames, the particles going out of boud. Consider a particle in frame 1 at the end, this particle exits the boundary in the next frame causing the calculations to malfunction as that particle will be forced to match with another particle. 

## Approach 3
To overcome this problem, we change the problem of matching particles to matching field. We visualize the particles as particle fields instead of discrete particles. We propagate the field according to the velocity predicted by the neural network and try to match the particle fields instead.

The code for this approach cannot be uploaded as of now.


