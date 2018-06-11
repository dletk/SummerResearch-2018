# %%
import cv2 as cv


# %% Load the model
model = cv.dnn.readNetFromTensorflow("./model.pb")

print(model)
