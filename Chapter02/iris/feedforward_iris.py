import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from ch02.util.matplot_util import draw_plot

RANDOMSEED = 40
tf.set_random_seed(RANDOMSEED)

def load_iris_data():

    from numpy import genfromtxt
    data = genfromtxt('iris.csv', delimiter=',')

    target = genfromtxt('target.csv', delimiter=',').astype(int)

    # Prepend the column of 1s for bias
    L, W  = data.shape
    all_X = np.ones((L, W + 1))
    all_X[:, 1:] = data
    num_labels = len(np.unique(target))
    all_y = np.eye(num_labels)[target]
    return train_test_split(all_X, all_y, test_size=0.33, random_state=RANDOMSEED)

def initialize_weights(shape, stddev):
    weights = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(weights)

def forward_propagation(X, weights_1, weights_2):
    sigmoid = tf.nn.sigmoid(tf.matmul(X, weights_1))
    y = tf.matmul(sigmoid, weights_2)
    return y

def run(h_size, stddev, sgd_step):
    train_x, test_x, train_y, test_y = load_iris_data()

    # Size of Layers
    x_size = train_x.shape[1]  # Input nodes: 4 features and 1 bias
    y_size = train_y.shape[1]  # Outcomes (3 iris flowers)

    # variables
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])
    weights_1 = initialize_weights((x_size, h_size), stddev)
    weights_2 = initialize_weights((h_size, y_size), stddev)

    y_pred = forward_propagation(X, weights_1, weights_2)
    predict = tf.argmax(y_pred, dimension=1)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
    updates_sgd = tf.train.GradientDescentOptimizer(sgd_step).minimize(cost)
    sess = tf.Session()
    # init = tf.global_variables_initializer()
    init = tf.initialize_all_variables()
    steps = 50
    sess.run(init)
    x  = np.arange(steps)
    test_acc = []
    train_acc = []
    print("Step, train accuracy, test accuracy")
    for step in range(steps):
        # Train with each example
        for i in range(len(train_x)):
            sess.run(updates_sgd, feed_dict={X: train_x[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_x, y: train_y}))
        test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                sess.run(predict, feed_dict={X: test_x, y: test_y}))

        print("%d, %.2f%%, %.2f%%"
              % (step + 1, 100. * train_accuracy, 100. * test_accuracy))
        #x.append(step)
        test_acc.append(100. * test_accuracy)
        train_acc.append(100. * train_accuracy)

    t = [np.array(test_acc)]
    t.append(train_acc)
    title = "Steps vs Accuracy-No of hidden nodes: " + str(h_size) + ", sgd step:" + str(sgd_step) + \
            ", std dev:" + str(stddev)
    label = ['Test Accuracy', 'Train Accuracy']
    draw_plot(x, t, title, label)
    sess.close()


def main():
    run(128,0.1,0.01)


if __name__ == '__main__':
    main()