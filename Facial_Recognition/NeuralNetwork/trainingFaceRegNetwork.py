# %% Import libraries
import numpy as np
import re
import cv2 as cv
import glob
import random

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3

from tensorflow import keras
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=config))

# %% ===========================================
# Load the images into numpy array, looping through all of the image using glob
# The directory containing all the images
parent_directory = "../alignedImages/data/faceScrub/*"
# Construct the images path list in sorted order
image_paths = sorted(glob.glob(parent_directory))
# random.shuffle(image_paths)
# print(image_paths)

# %% ===========================================
# Create a nparray containing all images
# Create a list to store raw image
# Also, create the label list as well
list_raw_images = []
labels_raw_images = []
last_label = ""
current_label = -1
labels_file = open("labels.txt", "w")
for path in image_paths:
    # Get the image
    image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    list_raw_images.append(image)

    # Prepare the label for the current image
    # In the faceScrub database, the first 2 word make up the label
    file_names = path.split()[:2]
    individial_name = file_names[0] + file_names[1]
    # If the name is the different with the last name, we are switching to a new individial
    if individial_name != last_label:
        current_label += 1
        last_label = individial_name
        labels_file.write(last_label+"\n")
    labels_raw_images.append(current_label)
labels_file.close()

# Create data and validation data
# Create a set of index to take away from data as validation
validation_index = set()
set_labels = set(labels_raw_images)

# Creating validation data by looping through the main data and take out elements at random indices
validation_data = []
validation_labels = []

for label in set_labels:
	label_indices = [i for i, val in enumerate(labels_raw_images) if val==label]
	# Take 3 images out to use as validation set
	for i in range(3):
		index = random.choice(label_indices)
		validation_data.append(list_raw_images[index])
		validation_labels.append(label)
		validation_index.add(index)

# Prepare the train data by excluding the validation data
data = []
labels = []
for i in range(len(list_raw_images)):
    if i not in validation_index:
        data.append(list_raw_images[i])
        labels.append(labels_raw_images[i])

# Convert all the array to nparray
data = np.asarray(data)
labels = np.asarray(labels)
validation_data = np.asarray(validation_data)
validation_labels = np.asarray(validation_labels)

# %% ===========================================
# Check the input
print(data.shape)
print(labels.shape)
print(validation_data.shape)
print(validation_labels.shape)

# %% ===========================================
# Transform the shape of the input data to (size, height, width, 1)
data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
validation_data = validation_data.reshape(validation_data.shape[0], validation_data.shape[1], validation_data.shape[2], 1)
# Transform the labels array into categorical
labels = keras.utils.to_categorical(labels)
validation_labels = keras.utils.to_categorical(validation_labels)
# Get the total number of categories in the data
numLabels = labels.shape[1]

# %% ===========================================
# Check the transformation
print(data.shape)
print(labels.shape)
print(numLabels)

# CREATING MODEL

# %% ==============================================
# Creating a model
model = keras.models.Sequential()


# %% ==============================================
# Add to layers to model
# The input shape must be declared for the first layer (the layter RIGHT AFTER the input)
model.add(keras.layers.Conv2D(filters=256, kernel_size=(5,5), padding="valid", activation="relu", input_shape=(192,176,1)))
# After doing convolution, using a max pooling layer to reduce the size
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
# Added dropout layers to avoid overfitting
model.add(keras.layers.Dropout(0.3))

# Second layer
model.add(keras.layers.Conv2D(filters=128, kernel_size=(4,4), activation="relu", padding="valid"))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
# Added dropout
model.add(keras.layers.Dropout(0.4))

# Third layer
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="valid"))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))

# Add a flatten layer to flat the input before feed into Dense layer
model.add(keras.layers.Flatten())

# Using the dense layers to analyze and produce out come
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(512, activation="relu"))
# model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dense(128, activation="relu"))

# Output the layer
model.add(keras.layers.Dense(numLabels, activation="softmax"))
# =============
# Print out the info of model
model.summary()

# SETTING UP OTHER PARAMETERS FOR THE TRAINING

# %% ==============================================
# Compile the model
model.compile(optimizer=keras.optimizers.SGD(0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# %% ==============================================
# Prepare the model check point to save the model after every epoch
checker = keras.callbacks.ModelCheckpoint("neuralNetworkFaceReg.h5", save_best_only=True, monitor="val_acc")

# %% ==============================================
# Train the model
print(labels.shape)
print(validation_labels.shape)
model.fit(data, labels, batch_size=32, epochs=1000, callbacks=[checker], validation_data=(validation_data, validation_labels))
