
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


def main():
    ops.reset_default_graph()

    session = tf.Session()

    x_val = np.linspace(start=-10., stop=10., num=1000)

    # ReLU activation
    y_relu = session.run(tf.nn.relu(x_val))

    # ReLU-6 activation
    y_relu6 = session.run(tf.nn.relu6(x_val))

    # Sigmoid activation
    y_sigmoid = session.run(tf.nn.sigmoid(x_val))

    # Hyper Tangent activation
    y_tanh = session.run(tf.nn.tanh(x_val))

    # Softsign activation
    y_softsign = session.run(tf.nn.softsign(x_val))

    # Softplus activation
    y_softplus = session.run(tf.nn.softplus(x_val))

    # Exponential linear activation
    print(session.run(tf.nn.elu([-1., 0., 1.])))
    y_elu = session.run(tf.nn.elu(x_val))


    plt.plot(x_val, y_softplus, 'r--', label='Softplus', linewidth=2)
    plt.plot(x_val, y_relu, 'b:', label='RELU', linewidth=2)
    plt.plot(x_val, y_relu6, 'g-.', label='RELU6', linewidth=2)
    plt.plot(x_val, y_elu, 'k-', label='ELU', linewidth=1)
    plt.ylim([-1.5,7])
    plt.legend(loc='top left')
    plt.title('Activation functions', y=1.05)
    plt.show()

    plt.plot(x_val, y_sigmoid, 'r--', label='Sigmoid', linewidth=2)
    plt.plot(x_val, y_tanh, 'b:', label='tanh', linewidth=2)
    plt.plot(x_val, y_softsign, 'g-.', label='Softsign', linewidth=2)

    plt.ylim([-1.5,1.5])
    plt.legend(loc='top left')
    plt.title('Activation functions with Vanishing Gradient', y=1.05)
    plt.show()

if __name__ == '__main__':
  main()