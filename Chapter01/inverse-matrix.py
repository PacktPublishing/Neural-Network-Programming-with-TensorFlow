import tensorflow as tf

mat = tf.placeholder(tf.float32)

matrix = [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
inv_mat = tf.matrix_inverse(mat)

with tf.Session() as sess:
    print(sess.run(inv_mat, feed_dict={mat: matrix}))