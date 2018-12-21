import subprocess
# manually put back imported modules
import tempfile
from pathlib import Path

import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util

tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess


def convert_model(sess, model, out_name: str, input_shape, out_format=None, num_outputs=1, quantize=False):
    if out_format is None:
        out_format = ['pb', 'pbtext', 'tflite']

    netin = tf.placeholder(name="img", dtype=tf.float32, shape=input_shape)

    num_output = num_outputs
    netout = [None] * num_output
    netout_names = [None] * num_output
    for i in range(num_output):
        netout_names[i] = str(i)
        netout[i] = tf.identity(model.outputs[i], name=netout_names[i])
    print('output nodes names are: ', netout_names)

    """
    Convert variables to constants
    """
    if quantize:
        from tensorflow.tools.graph_transforms import TransformGraph

        transforms = ["quantize_weights", "quantize_nodes"]
        transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], netout_names, transforms)
        constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, netout_names)
    else:
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), netout_names)

    """
    Save pb graph
    """
    if 'pb' in out_format:
        graph_io.write_graph(graph_or_graph_def=constant_graph, logdir='', name=out_name + '.pb', as_text=False)
        print('Saved the freezed graph (ready for inference) at: ', str(Path(out_name + '.pb')))
    if 'pbtext' in out_format:
        graph_io.write_graph(graph_or_graph_def=constant_graph, logdir='', name=out_name + '.pbtext', as_text=True)
        print('Saved the freezed graph (ready for inference) at: ', str(Path(out_name + '.pbtext')))

    """
    Convert to TFLite
    """
    if 'tflite' in out_format:
        tflite_model = tf.contrib.lite.toco_convert(constant_graph, [netin], netout)
        open(out_name + '.tflite', "wb").write(tflite_model)
        print('Saved tflite model at: ', str(Path(out_name + '.tflite')))
