# Summer research 2018

## Facial recognition using neural network on the NVIDIA Jetson TX2

### Current approach

#### Building a neural network using HOG features descriptors as input
1. Create a dataset of HOG features from facial images (using openCV and CUDA library)
2. Construct the neural network as followed:
* Create a convolutional neural network trained by HOG features of images
* Train the network to find X distinct features of a face.
* Output an array containing values of these features for a given face

#### Using SVM machine learning approach to match the face with the current database.
