import numpy as np
import tensorflow as tf

path = "/Users/manpreet.singh/Sandbox/codehub/github/science/neuralnetwork-programming/ch01/plots"
logs = "/Users/manpreet.singh/Sandbox/codehub/github/science/neuralnetwork-programming/ch01/logs"

xMatrix = np.array([[0,2,1,0,0,0,0,0],
              [2,0,0,1,0,1,0,0],
              [1,0,0,0,0,0,1,0],
              [0,1,0,0,1,0,0,0],
              [0,0,0,1,0,0,0,1],
              [0,1,0,0,0,0,0,1],
              [0,0,1,0,0,0,0,1],
              [0,0,0,0,1,1,1,0]], dtype=np.float32)


def pca(mat):
    mat = tf.constant(mat, dtype=tf.float32)
    mean = tf.reduce_mean(mat, 0)
    less = mat - mean
    s, u, v = tf.svd(less, full_matrices=True, compute_uv=True)

    s2 = s ** 2
    variance_ratio = s2 / tf.reduce_sum(s2)

    with tf.Session() as session:
        run = session.run([variance_ratio])
    return run


if __name__ == '__main__':
    print(pca(xMatrix))

