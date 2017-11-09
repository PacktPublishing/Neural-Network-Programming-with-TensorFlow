import tensorflow as tf


x = [[1,2,3],[4,5,6]]
x = tf.convert_to_tensor(x)
xtrans = tf.transpose(x)

y=([[[1,2,3],[6,5,4]],[[4,5,6],[3,6,3]]])
y = tf.convert_to_tensor(y)
ytrans = tf.transpose(y, perm=[0, 2, 1])

with tf.Session() as sess:
    print(sess.run(xtrans))
    print(sess.run(ytrans))