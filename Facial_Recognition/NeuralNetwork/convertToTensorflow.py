from tensorflow.python.keras import backend as K
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import sys

# Load the pretrained model in (.h5 file)
model = keras.models.load_model(sys.argv[1])
# Set the learning phase to 0 to able to convert and import to openCV
K.set_learning_phase(0)

# =================================================================
# ============== Drop the last layer of the model =================
#==================================================================
# We are interested in the secon last layer of the model for 128 measurements
model.pop()

print(model.summary())


# =================================================================
# ============ Convert the model to tensorflow model ==============
#==================================================================
pred_node_names = [None]
pred = [None]
for i in range(1):
    pred_node_names[i] = "output_node"+str(i)
    pred[i] = tf.identity(model.outputs[i], name=pred_node_names[i])

sess = K.get_session()
# Convert to a constant graph
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, ".", "model.pb", as_text=False)
