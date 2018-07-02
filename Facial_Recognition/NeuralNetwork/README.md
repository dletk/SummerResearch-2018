# Neural network design
## General problem

In the current approach of our facial recognition system, the neural network produces an encoded measurement of 128 distinct values for a particular face.
However, such a neural network is hard to design and train, since we do not possess the desired "target" to train the network. In fact, we do
not even know what are those 128 measurements. The current design is a work around for this problem.

## The network design

Based on our target, the neural network needs to produce 128 distinct measurements for a particular face. Therefore, it is fair to say that these measurements
can be used as the input for a classification layer of the neural network. We create a neural network with the last layer to be a classification layer as a normal
classifying problem. However, the second last layer will be a Dense layer with 128 nodes. Because we have the labels of all images in the dataset, this would be an 
easy design. After training this network, the last layer is dropped and the second last layer will be the output layer. If the network are highly accurate, then the
second last layer should produce unique value for each label, i.e., 128 distinct measurements.

## Network structure

At the moment, the best accuracy is 75% resulted from the following structure:

|Type | Number of filters/node | Parameters |
|:----|-----------------------:|-----------:|
|Convolutional 2D | 256 | kernel_size=(5,5)| 
|MaxPooling | NA | pool_size=(3,3) |
|Dropout| NA | 0.3 |
|Convolutional 2D | 128 | kernel_size=(4,4)| 
|MaxPooling | NA | pool_size=(3,3) |
|Dropout| NA | 0.4 |
|Convolutional 2D | 64 | kernel_size=(3,3)| 
|MaxPooling | NA | pool_size=(3,3) |
|Flatten | NA | NA |
|Dense | 512 | activation="relu" |
|Dense | 128 | activation="relu" |
|Dense | num_labels | activation="softmax" |

## Notice on the network structure

Because openCV deep neural network does not support tensorflow dropout layer at the time of writing, the use of this type of layer was avoided for the first trainings. However, the network nevers exceeds 60% accuracy with this setting. It is also worth noticing that the network learn the training data very well in this setting with 99% accuracy after less than 100 epochs. This indicates the effect of overfitting. Adding the dropout layers into the network seemed to solve this problem and improved the validation accuracy to 65% and then 75% rapidly. The network learn the training data in slower speed, making it extract more information and features to improve the validation accuracy along the way.

## HOW TO LOAD MODEL THE TRAINED MODEL FROM KERAS TO OPENCV

OpenCV currently does not support the functionality to load a Keras (h5) model directly to its neural network framework (at the time of writing). However, it does support loading TensorFlow model and making prediction. However, not all type of layers in TensorFlow and Keras are supported. Therefore, there are some prerequisites and preparation to modify the pretrained Keras model. Follow the below instructions **CAREFULLY**:
- TODO: All of these steps can be put into a bash script.
- Run the script convertToTensorflow.py to convert a Keras model to a TensorFlow model
```
python3 convertToTensorflow.py file/path/to/model.pb
```
- Run the following code (provided by Tensorflow) to optimize the model. The name of input and output can be different depending on the structure of the network. Check the model.pb to make sure.
```
python3 /usr/local/lib/python3.5/dist-packages/tensorflow/python/tools/optimize_for_inference.py   --input model.pb   --output opt_model.pb   --input_names conv2d_1_input   --output_names dense_3/Softmax
```
- Run the script makeTextGraphRepresentation.py to export the model.pb into a model.pbtxt file. This file represent the model structure, so we can perform manual modification on the model by changing this file.
```
python3 makeTextGraphRepresentation.py
```
- **Important** Delete all the drop out layers. Change the input of the layer right after the last drop out to the one currently before it (after all the deletions).
- **Important** Delete all the Flatten layers except the last one "Reshape". Delete all the "flatten" inputs that it has as well. Change its op from "Reshape" to "Flatten".
- The reference for these 2 steps can be found [here](http://answers.opencv.org/question/183682/opencv-dnn-import-dropout-layer-error-after-finetuning-keras-vgg16/).
- Should be good to go!!!
TODO: All of the 
