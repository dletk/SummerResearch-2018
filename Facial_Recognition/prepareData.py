import cv2 as cv
import glob
import os

# %% Load in the labels of the image
with open("./Facial_Recognition/list_name.txt") as labelsFile:
    labels = [line.strip() for line in labelsFile.readlines()]
print(labels)

# %% Load the data
# Set the parent directory for the raw data files
parent_directory  = "./Facial_Recognition/data/cfp-dataset/"
# Get all the images in the Images directory, including frontal and profile images
list_of_all_images = glob.glob(parent_directory+"/Images/*/*/*")
list_of_all_images.sort()

print(list_of_all_images)

print("===> Read images' path completed")

# Create a list of only frontal images
image_paths = []
for image_path in list_of_all_images:
    if "frontal" in image_path:
        image_paths.append(image_path)

print("===> Read frontal faces completed")

# Debug purpose
# print(image_paths)

# %% Create the database
# Create a new directory for exported data
export_directory = os.path.join(parent_directory, "data")

# Get size of frontal list
num_faces = len(image_paths)

# Loop through all the frontal images and prepare them for the database
index_image = 0;
for image_path in image_paths:
    # Create an image from the current image path
    image = cv.imread(image_path)

    # Resize that image to the standard size used in our database
    resizedImage = cv.resize(image, (176,192))

    # Export the resized image to the data folder
    cv.imwrite(os.path.join(export_directory, labels[index_image // 10] + " " + str(index_image)+".jpg"), resizedImage)

    print("--- Finished " + str(index_image) + " out of " + str(num_faces))
    index_image += 1

print("===> Exported completed")
