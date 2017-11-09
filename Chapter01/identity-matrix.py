import tensorflow as tf

identity = tf.eye(3, 3)

with tf.Session() as sess:
    print(sess.run(identity))