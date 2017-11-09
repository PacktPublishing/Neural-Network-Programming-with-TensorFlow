from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
define all the constants
"""
numEpochs = 10
seriesLength = 50000
backpropagationLength = 15
stateSize = 4
numClasses = 2
echoStep = 3
batchSize = 5
num_batches = seriesLength // batchSize // backpropagationLength


"""
generate data
"""
def generateData():
    x = np.array(np.random.choice(2, seriesLength, p=[0.5, 0.5]))
    y = np.roll(x, echoStep)
    y[0:echoStep] = 0

    x = x.reshape((batchSize, -1))
    y = y.reshape((batchSize, -1))

    return (x, y)


"""
start computational graph
"""
batchXHolder = tf.placeholder(tf.float32, [batchSize, backpropagationLength], name="x_input")
batchYHolder = tf.placeholder(tf.int32, [batchSize, backpropagationLength], name="y_input")

initState = tf.placeholder(tf.float32, [batchSize, stateSize], "rnn_init_state")

W = tf.Variable(np.random.rand(stateSize+1, stateSize), dtype=tf.float32, name="weight1")
bias1 = tf.Variable(np.zeros((1,stateSize)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(stateSize, numClasses),dtype=tf.float32, name="weight2")
bias2 = tf.Variable(np.zeros((1,numClasses)), dtype=tf.float32)

tf.summary.histogram(name="weights", values=W)


# Unpack columns
inputsSeries = tf.split(axis=1, num_or_size_splits=backpropagationLength, value=batchXHolder)
labelsSeries = tf.unstack(batchYHolder, axis=1)

# Forward passes
from tensorflow.contrib import rnn
cell = rnn.BasicRNNCell(stateSize)
statesSeries, currentState = rnn.static_rnn(cell, inputsSeries, initState)

# calculate loss
logits_series = [tf.matmul(state, W2) + bias2 for state in statesSeries]
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits) for logits, labels in zip(logits_series,labelsSeries)]
total_loss = tf.reduce_mean(losses, name="total_loss")

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss, name="training")


"""
plot computation
"""
def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batchSeriesIdx in range(5):
        oneHotOutputSeries = np.array(predictions_series)[:, batchSeriesIdx, :]
        singleOutputSeries = np.array([(1 if out[0] < 0.5 else 0) for out in oneHotOutputSeries])

        plt.subplot(2, 3, batchSeriesIdx + 2)
        plt.cla()
        plt.axis([0, backpropagationLength, 0, 2])
        left_offset = range(backpropagationLength)
        plt.bar(left_offset, batchX[batchSeriesIdx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batchSeriesIdx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, singleOutputSeries * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)


"""
run the graph
"""
with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs", graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(numEpochs):
        x,y = generateData()
        _current_state = np.zeros((batchSize, stateSize))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * backpropagationLength
            end_idx = start_idx + backpropagationLength

            batchX = x[:,start_idx:end_idx]
            batchY = y[:,start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, currentState, predictions_series],
                feed_dict={
                    batchXHolder:batchX,
                    batchYHolder:batchY,
                    initState:_current_state
                })

            loss_list.append(_total_loss)

            # fix the cost summary later
            tf.summary.scalar(name="totalloss", tensor=_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()