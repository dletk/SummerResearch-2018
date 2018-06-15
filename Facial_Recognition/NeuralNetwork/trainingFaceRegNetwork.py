# %% Import libraries
import numpy as np
import re
import cv2 as cv
import glob
import random

from tensorflow import keras

# %% ===========================================
# Load the images into numpy array, looping through all of the image using glob
# The directory containing all the images
parent_directory = "../alignedImages/data/faceScrub/*"
# Construct the images path list, the list is in random order
image_paths = sorted(glob.glob(parent_directory))
# random.shuffle(image_paths)
# print(image_paths)

# %% ===========================================
# Create a nparray containing all images
# Create a list to store raw image
# Also, create the label list as well
list_raw_images = []
labels = []
last_label = ""
current_label = 0
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
    labels.append(current_label)

# Create a numpy from list
list_images = np.asarray(list_raw_images)
labels = np.asarray(labels)

# %% ===========================================
# Check the input
print(list_images.shape)
print(labels.shape)

# %% ===========================================
# Transform the shape of the input data to (size, height, width, 1)
list_images = list_images.reshape(list_images.shape[0], list_images.shape[1], list_images.shape[2], 1)
# Transform the labels array into categorical
labels = keras.utils.to_categorical(labels)
# Get the total number of categories in the data
numLabels = labels.shape[1]

# %% ===========================================
# Check the transformation
print(list_images.shape)
print(labels.shape)
print(numLabels)

# CREATING MODEL

# %% ==============================================
# Creating a model
model = keras.models.Sequential()


# %% ==============================================
# Add to layers to model
# The input shape must be declared for the first layer (the layter RIGHT AFTER the input)
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="valid", activation="relu", input_shape=(192,176,1)))
# After doing convolution, using a max pooling layer to reduce the size
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Second layer
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="valid"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Add a flatten layer to flat the input before feed into Dense layer
model.add(keras.layers.Flatten())

# Using the dense layers to analyze and produce out come
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dense(128, activation="relu"))

# Output the layer
model.add(keras.layers.Dense(numLabels, activation="softmax"))
# =============
# Print out the info of model
model.summary()

# SETTING UP OTHER PARAMETERS FOR THE TRAINING

# %% ==============================================
# Compile the model
model.compile(optimizer=keras.optimizers.SGD(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# %% ==============================================
# Prepare the model check point to save the model after every epoch
checker = keras.callbacks.ModelCheckpoint("neuralNetworkFaceReg.h5", save_best_only=True, monitor="loss")

# %% ==============================================
# Train the model
model.fit(list_images, labels, batch_size=16, epochs=100, callbacks=[checker])
