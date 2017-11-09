import tensorflow as tf

vector = tf.constant([[4,5,6]], dtype=tf.float32)

eucNorm = tf.norm(vector, ord="euclidean")

with tf.Session() as sess:
    print(sess.run(eucNorm))