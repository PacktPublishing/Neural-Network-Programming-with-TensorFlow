import tensorflow as tf

mat1 = tf.constant([[4, 5, 6],[3,2,1]])
mat2 = tf.constant([[7, 8, 9],[10, 11, 12]])

# hadamard product (element wise)
mult = tf.multiply(mat1, mat2)

# dot product (no. of rows = no. of columns)
dotprod = tf.matmul(mat1, tf.transpose(mat2))

with tf.Session() as sess:
    print(sess.run(mult))
    print(sess.run(dotprod))