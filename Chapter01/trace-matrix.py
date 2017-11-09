import tensorflow as tf

mat = tf.constant([
 [0, 1, 2],
 [3, 4, 5],
 [6, 7, 8]
], dtype=tf.float32)

# get trace ('sum of diagonal elements') of the matrix
mat = tf.trace(mat)

with tf.Session() as sess:
    print(sess.run(mat))