import tensorflow as tf

from utils.workdir import cd_work

cd_work()

model_filename = 'logs/gatev8_mixed/gatev8.pb'
log_dir = '/home/phil/Downloads/temp/'

with tf.Session() as sess:
    with tf.gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)

train_writer = tf.summary.FileWriter(log_dir)
train_writer.add_graph(sess.graph)

train_writer.flush()
train_writer.close()
