# %% Import required libraries
import time
import keras
import tensorflow as tf

# %% Load the data into
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# %% Normalizing the image on the scale of 0-255
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# %% Change the shape of the data to 28,28,1
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# %% Check data
y_train.shape
x_train.shape

# %% Create the model
model = keras.models.Sequential()

# %% Add to layers to model
# The input shape must be declared for the first layer (the layter RIGHT AFTER the input)
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", input_shape=(28,28,1)))

# After doing convolution, using a max pooling layer to reduce the size
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
# Drop out some input to avoid overfitting
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.2))

# Add a flatten layer to flat the input before feed into Dense layer
model.add(keras.layers.Flatten())

# Using the dense layers to analyze and produce out come
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(10, activation="softmax"))

# Print out the info of model
model.summary()

# %% Function to export the model to frozen_model for tensorflow
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

# %% Load the model from disk
with open("./Facial_Recognition/NeuralNetwork/fashion_mnistModel.json", "r") as json_file:
    model = keras.models.model_from_json(json_file.read())
model.load_weights("./Facial_Recognition/NeuralNetwork/fashion_mnistWeight.h5")

# %% Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# %% Prepare the early stopping
early_stop = keras.callbacks.EarlyStopping(patience=2)

# %% Train the model
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2, callbacks=[early_stop])

# %% Test the model after training
model.evaluate(x_test, y_test)
# %% Predict
model.predict(x_test)

# %% Save the model
# Save the structure
modelJSON = model.to_json()
with open("fashion_mnistModel.json", "w") as json_file:
    json_file.write(modelJSON)

# Save the weight
model.save_weights("./fashion_mnistWeight.h5")


# %% Create the fronzen model from the current model
frozen_graph = freeze_session(keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, "./", "fashionMNISTmodel.pb", as_text=False)
