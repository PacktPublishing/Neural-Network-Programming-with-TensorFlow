import tensorflow as tf
import numpy as np

X = tf.Variable(np.random.random_sample(), dtype=tf.float32)
y = tf.Variable(np.random.random_sample(), dtype=tf.float32)


def createCons(x):
    return tf.constant(x, dtype=tf.float32)

function = tf.pow(X, createCons(2)) + createCons(2) * X * y + createCons(3) * tf.pow(y, createCons(2)) + createCons(4) * X + createCons(5) * y + createCons(6)


# compute hessian
def hessian(func, varbles):
    matrix = []
    for v_1 in varbles:
        tmp = []
        for v_2 in varbles:
            # calculate derivative twice, first w.r.t v2 and then w.r.t v1
            tmp.append(tf.gradients(tf.gradients(func, v_2)[0], v_1)[0])
        tmp = [createCons(0) if t == None else t for t in tmp]
        tmp = tf.stack(tmp)
        matrix.append(tmp)
    matrix = tf.stack(matrix)
    return matrix

hessian = hessian(function, [X, y])

sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(hessian))


