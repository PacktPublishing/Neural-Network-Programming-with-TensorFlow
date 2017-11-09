import tensorflow as tf
import numpy as np

x = np.array([[10.0, 15.0, 20.0], [0.0, 1.0, 5.0], [3.0, 5.0, 7.0]], dtype=np.float32)

det = tf.matrix_determinant(x)

with tf.Session() as sess:
    print(sess.run(det))