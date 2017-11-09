import numpy as np
import tensorflow as tf
from string import punctuation
from collections import Counter


'''
movie review dataset for sentiment analysis
'''
with open('data/reviews.txt', 'r') as f:
    movieReviews = f.read()
with open('data/labels.txt', 'r') as f:
    labels = f.read()


'''
data cleansing - remove punctuations
'''
text = ''.join([c for c in movieReviews if c not in punctuation])
movieReviews = text.split('\n')

text = ' '.join(movieReviews)
words = text.split()

print(text[:500])
print(words[:100])


'''
build a dictionary that maps words to integers
'''
counts = Counter(words)
vocabulary = sorted(counts, key=counts.get, reverse=True)
vocab2int = {word: i for i, word in enumerate(vocabulary, 1)}

reviewsInts = []
for review in movieReviews:
    reviewsInts.append([vocab2int[word] for word in review.split()])


'''
convert labels from positive and negative to 1 and 0 respectively
'''
labels = labels.split('\n')
labels = np.array([1 if label == 'positive' else 0 for label in labels])

reviewLengths = Counter([len(x) for x in reviewsInts])
print("Min review length are: {}".format(reviewLengths[0]))
print("Maximum review length are: {}".format(max(reviewLengths)))


'''
remove the review with zero length from the reviewsInts list
'''
nonZeroIndex = [i for i, review in enumerate(reviewsInts) if len(review) != 0]
print(len(nonZeroIndex))


'''
turns out its the final review that has zero length. But that might not always be the case, so let's make it more
general.
'''
reviewsInts = [reviewsInts[i] for i in nonZeroIndex]
labels = np.array([labels[i] for i in nonZeroIndex])


'''
create an array features that contains the data we'll pass to the network. The data should come from reviewInts, since
we want to feed integers to the network. Each row should be 200 elements long. For reviews shorter than 200 words, 
left pad with 0s. That is, if the review is ['best', 'movie', 'renaira'], [100, 40, 20] as integers, the row will look 
like [0, 0, 0, ..., 0, 100, 40, 20]. For reviews longer than 200, use on the first 200 words as the feature vector.
'''
seqLen = 200
features = np.zeros((len(reviewsInts), seqLen), dtype=int)
for i, row in enumerate(reviewsInts):
    features[i, -len(row):] = np.array(row)[:seqLen]

print(features[:10,:100])


'''
lets create training, validation and test data sets. trainX and trainY for example. 
also define a split percentage function 'splitPerc' as the percentage of data to keep in the training 
set. usually this is 0.8 or 0.9.
'''
splitPrec = 0.8
splitIndex = int(len(features)*0.8)
trainX, valX = features[:splitIndex], features[splitIndex:]
trainY, valY = labels[:splitIndex], labels[splitIndex:]

testIndex = int(len(valX)*0.5)
valX, testX = valX[:testIndex], valX[testIndex:]
valY, testY = valY[:testIndex], valY[testIndex:]

print("Train set: {}".format(trainX.shape), "\nValidation set: {}".format(valX.shape), "\nTest set: {}".format(testX.shape))
print("label set: {}".format(trainY.shape), "\nValidation label set: {}".format(valY.shape), "\nTest label set: {}".format(testY.shape))


'''
tensor-flow computational graph
'''
lstmSize = 256
lstmLayers = 1
batchSize = 500
learningRate = 0.001

nWords = len(vocab2int) + 1

# create graph object and add nodes to the graph
graph = tf.Graph()

with graph.as_default():
    inputData = tf.placeholder(tf.int32, [None, None], name='inputData')
    labels = tf.placeholder(tf.int32, [None, None], name='labels')
    keepProb = tf.placeholder(tf.float32, name='keepProb')


'''
let us create the embedding layer (word2vec)
'''
# number of neurons in hidden or embedding layer
embedSize = 300

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((nWords, embedSize), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputData)


'''
lets use tf.contrib.rnn.BasicLSTMCell to create an LSTM cell, later add drop out to it with 
tf.contrib.rnn.DropoutWrapper. and finally create multiple LSTM layers with tf.contrib.rnn.MultiRNNCell.
'''
with graph.as_default():
    with tf.name_scope("RNNLayers"):
        def createLSTMCell():
            lstm = tf.contrib.rnn.BasicLSTMCell(lstmSize, reuse=tf.get_variable_scope().reuse)
            return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keepProb)

        cell = tf.contrib.rnn.MultiRNNCell([createLSTMCell() for _ in range(lstmLayers)])

        initialState = cell.zero_state(batchSize, tf.float32)


'''
set tf.nn.dynamic_rnn to add the forward pass through the RNN. here we're actually passing in vectors from the 
embedding layer 'embed'.
'''
with graph.as_default():
    outputs, finalState = tf.nn.dynamic_rnn(cell, embed, initial_state=initialState)


'''
final output will carry the sentiment prediction, therefore lets get the last output with outputs[:, -1], 
the we calculate the cost from that and labels.
'''
with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels, predictions)

    optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)


'''
now we can add a few nodes to calculate the accuracy which we'll use in the validation pass.
'''
with graph.as_default():
    correctPred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels)
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

'''
get batches
'''


def getBatches(x, y, batchSize=100):
    nBatches = len(x) // batchSize
    x, y = x[:nBatches * batchSize], y[:nBatches * batchSize]
    for i in range(0, len(x), batchSize):
        yield x[i:i + batchSize], y[i:i + batchSize]


'''
training phase
'''
epochs = 1

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    writer = tf.summary.FileWriter("logs", graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initialState)

        for i, (x, y) in enumerate(getBatches(trainX, trainY, batchSize), 1):
            feed = {inputData: x, labels: y[:, None], keepProb: 0.5, initialState: state}

            loss, state, _ = sess.run([cost, finalState, optimizer], feed_dict=feed)

            if iteration % 5 == 0:
                print("Epoch are: {}/{}".format(e, epochs), "Iteration is: {}".format(iteration), "Train loss is: {:.3f}".format(loss))

            if iteration % 25 == 0:
                valAcc = []
                valState = sess.run(cell.zero_state(batchSize, tf.float32))
                for x, y in getBatches(valX, valY, batchSize):
                    feed = {inputData: x, labels: y[:, None], keepProb: 1, initialState: valState}
                    batchAcc, valState = sess.run([accuracy, finalState], feed_dict=feed)
                    valAcc.append(batchAcc)
                print("Val acc: {:.3f}".format(np.mean(valAcc)))
            iteration += 1
            saver.save(sess, "checkpoints/sentimentanalysis.ckpt")
    saver.save(sess, "checkpoints/sentimentanalysis.ckpt")

'''
testing phase
'''
testAcc = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, "checkpoints/sentiment.ckpt")

    testState = sess.run(cell.zero_state(batchSize, tf.float32))
    for i, (x, y) in enumerate(getBatches(testY, testY, batchSize), 1):
        feed = {inputData: x,
                labels: y[:, None],
                keepProb: 1,
                initialState: testState}
        batchAcc, testState = sess.run([accuracy, finalState], feed_dict=feed)
        testAcc.append(batchAcc)
    print("Test accuracy is: {:.3f}".format(np.mean(testAcc)))