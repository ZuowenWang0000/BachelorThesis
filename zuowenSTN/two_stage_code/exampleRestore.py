import tensorflow as tf
# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp
import tensorflow.contrib.image

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./checkpoint-80000.meta')
    saver.restore(sess, "./checkpoint-80000")
    # print all tensors in checkpoint file
    chkp.print_tensors_in_checkpoint_file("./checkpoint-80000",        tensor_name='', all_tensors=True)
