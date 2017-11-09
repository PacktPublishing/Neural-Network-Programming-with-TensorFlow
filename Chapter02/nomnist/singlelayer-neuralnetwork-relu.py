import cPickle as pickle

import numpy as np
import tensorflow as tf

from ch02.util.matplot_util import draw_plot
from util import accuracy

pickle_file = 'notMNIST.pickle'

image_size = 28
num_of_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_of_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

batch_size = 128

def main():
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        training_dataset = save['train_dataset']
        training_labels = save['train_labels']
        validation_dataset = save['valid_dataset']
        validation_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        print 'Training set', training_dataset.shape, training_labels.shape
        print 'Validation set', validation_dataset.shape, validation_labels.shape
        print 'Test set', test_dataset.shape, test_labels.shape

    train_dataset, train_labels = reformat(training_dataset, training_labels)
    valid_dataset, valid_labels = reformat(validation_dataset, validation_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print 'Training dataset shape', train_dataset.shape, train_labels.shape
    print 'Validation dataset shape', valid_dataset.shape, valid_labels.shape
    print 'Test dataset shape', test_dataset.shape, test_labels.shape

    graph = tf.Graph()
    no_of_neurons = 1024
    with graph.as_default():
        # Placeholder that will be fed
        # at run time with a training minibatch in the session
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_of_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        w1 = tf.Variable(
            tf.truncated_normal([image_size * image_size, no_of_neurons]))
        b1 = tf.Variable(tf.zeros([no_of_neurons]))

        w2 = tf.Variable(
            tf.truncated_normal([no_of_neurons, num_of_labels]))
        b2 = tf.Variable(tf.zeros([num_of_labels]))

        hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, w1) + b1)

        # Training computation.
        logits = tf.matmul(hidden1, w2) + b2
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, w1) + b1), w2) + b2)
        test_prediction = tf.nn.softmax(
            tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, w1) + b1), w2) + b2)

    num_steps = 101
    x = np.arange(num_steps)
    #x = []
    minibatch_acc = []
    validation_acc = []
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in xrange(num_steps):

            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            minibatch_accuracy = accuracy(predictions, batch_labels)
            validation_accuracy = accuracy(
                valid_prediction.eval(), valid_labels)

            if (step % 10 == 0):

                print("Minibatch loss at step", step, ":", l)
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % validation_accuracy)
            minibatch_acc.append( minibatch_accuracy)
            validation_acc.append( validation_accuracy)

            t = [np.array(minibatch_acc)]
            t.append(validation_acc)

        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
        title = "NotMNIST DataSet - Single Hidden Layer - 1024 neurons Activation function: RELU"
        label = ['Minibatch Accuracy', 'Validation Accuracy']
        draw_plot(x, t, title, label)

if __name__ == '__main__':
  main()