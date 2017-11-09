import tensorflow as tf
# k = tf.constant([
#     [1, 0, 1],
#     [2, 1, 0],
#     [0, 0, 1]
# ], dtype=tf.float32, name='k')
# i = tf.constant([
#     [4, 3, 1, 0],
#     [2, 1, 0, 1],
#     [1, 2, 4, 1],
#     [3, 1, 0, 2]
# ], dtype=tf.float32, name='i')
# kernel = tf.reshape(k, [3, 3, 1, 1], name='kernel')
# image  = tf.reshape(i, [1, 4, 4, 1], name='image')
# res = tf.squeeze(tf.nn.conv2d(image, kernel, [1, 1, 1, 1], "VALID"))
#
# with tf.Session() as sess:
#    print sess.run(res)
#

i = tf.constant([
                 [1.0, 1.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 1.0, 1.0],
                 [0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0]

        ], dtype=tf.float32)


k = tf.constant([
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0]
        ],  dtype=tf.float32),

kernel = tf.reshape(k, [3, 3, 1, 1], name='kernel')
image = tf.reshape(i, [1, 4, 5, 1], name='image')

res = tf.squeeze(tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding="VALID"))
# VALID means no padding
with tf.Session() as sess:
    print sess.run(res)