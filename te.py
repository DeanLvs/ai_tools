import os
import tensorflow as tf

print(tf.__version__)

# A simple TensorFlow operation to test
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
c = tf.matmul(a, b)

print(c)
