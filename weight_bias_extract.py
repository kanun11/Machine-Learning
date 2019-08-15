
import tensorflow as tf
import numpy as np




with tf.Session() as sess:
    new_saver1 = tf.train.import_meta_graph('/Users/kanun/Desktop/ML_project/python_code/tanh/tanh.meta')
    graph1=new_saver1.restore(sess, tf.train.latest_checkpoint('./tanh'))

    graph1 = tf.get_default_graph()

    predict1 = graph1.get_tensor_by_name('predict:0')
    w1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'w_hidden1')
    w2= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'w_hidden2')
    w_out = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'w_out')
    b1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'b_hidden1')
    b2= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'b_hidden2')
    b_out= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'b_out')

    value = sess.run(w1)
print(value)
