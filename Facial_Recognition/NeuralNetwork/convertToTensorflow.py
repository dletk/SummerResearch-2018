# %% Import libraries
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras import backend as K
import sys
import graph_util

# %% Load the model in from user input
model = keras.models.load_model(sys.argv[1])

# %% Export the graph compatible with cv2
# Serialize and fix the graph
sess = K.get_session()
graph_def = sess.graph.as_graph_def(add_shapes=True)
graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, [model.output.name.split(':')[0]])
graph_util.make_cv2_compatible(graph_def)

# %% Create the frozen model from the current model
tf.train.write_graph(graph_def, '', 'model.pbtxt', as_text=True)
