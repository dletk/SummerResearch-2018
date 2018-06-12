# %%
import cv2 as cv
import keras
import numpy as np

# %% Load the model
inp = np.random.random([1, 28, 28, 1])
inp = inp.astype("float32")
print(inp.shape)
in2 = inp.transpose(0, 3, 1, 2)
net = cv.dnn.readNetFromTensorflow("model.pb")

net.setInput(in2)
cv_out = net.forward()

print(cv_out)
