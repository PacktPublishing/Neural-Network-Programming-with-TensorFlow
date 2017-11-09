import tensorflow as tf


"""
try tf one hot encoder
"""
indices = [0, 2, -1, 1]
depth = 3
on_value = 5.0
off_value = 0.0
axis = -1

# 4*3 matrix
one_hot = tf.one_hot(indices, depth, on_value, off_value, axis)

with tf.Session() as sess:
    sess.run(tf.Print(one_hot, [one_hot]))

indices = [[0, 2], [1, -1]]
depth = 3
on_value = 1.0
off_value = 0.0
axis = -1

# 2*2*3 matrix
one_hot = tf.one_hot(indices, depth, on_value, off_value, axis)

with tf.Session() as sess:
    sess.run(tf.Print(one_hot, [one_hot]))

indices = [[0, 1, 1, 0, 1], [1, 1, 1, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1]]
depth = 2
on_value = 1.0
off_value = 0.0
axis = -1
one_hot = tf.one_hot(indices, depth, on_value, off_value, axis)
with tf.Session() as sess:
    sess.run(tf.Print(one_hot, [one_hot]))
    print(sess.run(tf.rank(one_hot)))
    print(sess.run(tf.shape(one_hot)))

"""
try tf rank, shape
"""
t = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
t1 = [[[1,2],[2,2],[3,2]], [[4,2],[5,2],[6,2]]]
ten = tf.convert_to_tensor(t1)

with tf.Session() as sess:
    print(sess.run(tf.rank(ten)))
    print(sess.run(tf.shape(ten)))


"""
try tf stack/unstack
"""
x = [1, 4]
y = [2, 5]
z = [3, 6]
with tf.Session() as sess:
    print(sess.run(tf.stack([x, y, z])))

    # Pack along first dim.
    stk = sess.run(tf.stack([x, y, z], axis=1))

    # unstack along 1st axis
    print(sess.run(tf.unstack(stk)))
    print(sess.run(tf.unstack(stk, axis=1)))