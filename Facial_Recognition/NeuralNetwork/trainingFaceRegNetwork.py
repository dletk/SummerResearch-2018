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
current_label = -1
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

# Create data and validation data
# Create a list of index to take away from data as validation, keep 5% as validation ( which means // 20)
validation_index = [random.randint(0, len(list_raw_images)-1) for x in range(len(list_raw_images) // 20)]
# Create temporary copy of labels and list_raw_images
temp_images = list_raw_images[:] # We have to use slice because python is pass py reference
temp_labels = labels[:]

# Creating validation data by looping through the main data and take out elements at given indices
validation_data = []
validation_labels = []
for index in validation_index:
    validation_data.append(temp_images[index])
    validation_labels.append(temp_labels[index])
    list_raw_images.pop(index)
    labels.pop(index)

data = np.asarray(list_raw_images)
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
checker = keras.callbacks.ModelCheckpoint("neuralNetworkFaceReg.h5", save_best_only=True, monitor="val_loss")

# %% ==============================================
# Train the model
model.fit(data, labels, batch_size=16, epochs=100, callbacks=[checker], validation_data=(validation_data, validation_labels))
