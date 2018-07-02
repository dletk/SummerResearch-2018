from tensorflow.python.keras import backend as K
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import sys

model = keras.models.load_model(sys.argv[1])
K.set_learning_phase(0)

pred_node_names = [None]
pred = [None]
for i in range(1):
    pred_node_names[i] = "output_node"+str(i)
    pred[i] = tf.identity(model.outputs[i], name=pred_node_names[i])

sess = K.get_session()
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, ".", "model.pb", as_text=False)
