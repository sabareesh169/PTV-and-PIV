# Physics informed Deep Learning to make velocity predictions

## Introduction
The purpose of this project is to find the fluid velocity from the images of a particle laden flow. There is a lot of research done into this problem which is called the Particle Image Velocimetry(PIV)/Particle Tracking Velocimetry(PTV) in Mechanical Engineering. The general methods followed for PIV is to break the field in frame 1 into interrogation windows and find the window with the highest correlation in the second frame to approximate the velocity at the position of interrogation window.

![General methodology followed for PIV](https://www.researchgate.net/profile/Daniel_Cote2/publication/50268148/figure/fig2/AS:305818365906946@1449924188734/The-underlying-concept-of-the-two-frames-cross-correlation-particle-image-velocimetry.png)

As we can guess, the window size is very critical for the measurement. Increasing the size of the window increases the accuracy of measurements but reduces the resolution as we wont get any finer details of the fluid flow within the window. Particle Tracking Velocimetry uses fewer particles to be able to increase the resolution of the predictions but becomes more and more difficult to track particles and the predictions are far and few in between. So, there is a tradeoff between accuracy and resolution in both cases. 

The objective of this project is to use AI to solve this tradeoff.

## Approach
The idea is to apply deep learning to make predictions. Given the velocity measurements from PIV, we can fit a Deep Neural Network on this data to get an enhanced resolution data of the fluid velocity. But the drawback of this approach is this curve fitting is that it tends to break some essential conditions like boundary conditions, initial conditions or the governing equations of the flow. 

![The inputs of the network time and position and the network spits out velocity in x and y directions](https://user-images.githubusercontent.com/25951391/78628223-67a3fc00-7848-11ea-9470-d5606d15df9b.png)

So, we need to implement the governing equations of the flow on the neural network as well. A Deep Neural Network performs backpropogation to change the weights of the network in order to minimize a prediction loss. In a normal regression modeling approach, the loss would be rmse between the actual values and the predictions (in our case between measurements from PIV/PTV and predictions of DNN). We now add the residual of the governing equations of the fluid to the loss function as well. For example, one of the governing equations we may want to implement is the 'conservation of mass' or 'continuity equation'. 
![The continutity equation](https://user-images.githubusercontent.com/25951391/78628442-f9ac0480-7848-11ea-95eb-0543ecddb8ff.png)
![The residual of the continuity equation](https://user-images.githubusercontent.com/25951391/78628443-fa449b00-7848-11ea-9918-e811f235eb63.png)
By adding the residual to the loss function, we minimize the residual which is effectively implementing the continuity equation on the predictions. 

