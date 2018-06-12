# %%
import cv2 as cv
import keras
import numpy as np

# %% Load the model
inp = np.random.random([1, 28, 28, 1]).astype("float32")
in2 = inp.transpose(0, 3, 1, 2)
net = cv.dnn.readNetFromTensorflow("model.pb")

print(in2)

net.setInput(in2)
cv_out = net.forward()

print(cv_out)
