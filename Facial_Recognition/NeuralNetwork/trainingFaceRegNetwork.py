# %% Import libraries
import numpy as np
import re
import cv2 as cv
import glob

import keras

# %% ===========================================
# Load the images into numpy array, looping through all of the image using glob
# The directory containing all the images
parent_directory = "../alignedImages/data/cfp-dataset/data/*"
# Construct the images path list, the list is in random order, so sort it as well
image_paths = sorted(glob.glob(parent_directory))
# print(image_paths)

# %% ===========================================
# Create a nparray containing all images
# Create a list to store raw image
# Also, create the label list as well
list_raw_images = []
labels = []
for path in image_paths:
    image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    list_raw_images.append(image)

    # Process to find label
    all_nums_in_path = re.findall(r"\d+", path)
    # The name of the file may contains some number other than the file index, so only take the last number to make sure
    file_index = all_nums_in_path[-1]
    # There are maximum 10 pics of a label, marked from 0-9, so ignore the last number is good enough
    fileLabel = int(file_index[:-1]) if len(file_index) > 1 else 0
    labels.append(fileLabel)

# Create a numpy from list
list_images = np.asarray(list_raw_images)
labels = np.asarray(labels)

# %% ===========================================
# Check the input
print(list_images.shape)
print(labels.shape)

# %% ===========================================
# Transform the shape of the input data to (size, height, width, 1)
list_images.reshape(list_images.shape[0], list_images.shape[1], list_images.shape[2], 1)
# Transform the labels array into categorical
labels = keras.utils.to_categorical(labels)

# %% ===========================================
# Check the transformation
list_images.shape
labels.shape
