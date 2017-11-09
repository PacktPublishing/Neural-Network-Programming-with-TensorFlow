import time
import tensorflow as tf
import numpy as np
import utility
from tqdm import tqdm
from urllib.request import urlretrieve
from os.path import isfile, isdir
import zipfile
from collections import Counter
import random

dataDir = 'data'
dataFile = 'text8.zip'
datasetName = 'text 8 data set'

'''
track progress of file download
'''


class DownloadProgress(tqdm):
    lastBlock = 0

    def hook(self, blockNum=1, blockSize=1, totalSize=None):
        self.total = totalSize
        self.update((blockNum - self.lastBlock) * blockSize)
        self.lastBlock = blockNum


if not isfile(dataFile):
    with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc=datasetName) as progressBar:
        urlretrieve('http://mattmahoney.net/dc/text8.zip', dataFile, progressBar.hook)

if not isdir(dataDir):
    with zipfile.ZipFile(dataFile) as zipRef:
        zipRef.extractall(dataDir)

with open('data/text8') as f:
    text = f.read()

'''
pre process the downloaded wiki text
'''
words = utility.preProcess(text)
print(words[:30])

print('Total words: {}'.format(len(words)))
print('Unique words: {}'.format(len(set(words))))

'''
convert words to integers
'''
int2vocab, vocab2int = utility.lookupTable(words)
intWords = [vocab2int[word] for word in words]
print('test')

'''
sub sampling (***think of words as int's***)
'''
threshold = 1e-5
wordCounts = Counter(intWords)
totalCount = len(intWords)
frequency = {word: count / totalCount for word, count in wordCounts.items()}
probOfWords = {word: 1 - np.sqrt(threshold / frequency[word]) for word in wordCounts}
trainWords = [word for word in intWords if random.random() < (1 - probOfWords[word])]

'''
get window batches
'''


def getTarget(words, index, windowSize=5):
    rNum = np.random.randint(1, windowSize + 1)
    start = index - rNum if (index - rNum) > 0 else 0
    stop = index + rNum
    targetWords = set(words[start:index] + words[index + 1:stop + 1])

    return list(targetWords)


'''
Create a generator of word batches as a tuple (inputs, targets)
'''


def getBatches(words, batchSize, windowSize=5):
    nBatches = len(words) // batchSize
    print('no. of batches {}'.format(nBatches))

    # only full batches
    words = words[:nBatches * batchSize]

    start = 0
    for index in range(0, len(words), batchSize):
        x = []
        y = []
        stop = start + batchSize
        batchWords = words[start:stop]
        for idx in range(0, len(batchWords), 1):
            yBatch = getTarget(batchWords, idx, windowSize)
            y.extend(yBatch)
            x.extend([batchWords[idx]] * len(yBatch))
        start = stop + 1
        yield x, y


'''
start computational graph
'''
train_graph = tf.Graph()
with train_graph.as_default():
    netInputs = tf.placeholder(tf.int32, [None], name='inputS')
    netLabels = tf.placeholder(tf.int32, [None, None], name='labelS')


'''
create embedding layer
'''
nVocab = len(int2vocab)
nEmbedding = 300
with train_graph.as_default():
    embedding = tf.Variable(tf.random_uniform((nVocab, nEmbedding), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, netInputs)


'''
Below, create weights and biases for the softmax layer. Then, use tf.nn.sampled_softmax_loss to calculate the loss
'''
n_sampled = 100
with train_graph.as_default():
    soft_W = tf.Variable(tf.truncated_normal((nVocab, nEmbedding)))
    soft_b = tf.Variable(tf.zeros(nVocab), name="softmax_bias")

    # Calculate the loss using negative sampling
    loss = tf.nn.sampled_softmax_loss(
        weights=soft_W,
        biases=soft_b,
        labels=netLabels,
        inputs=embed,
        num_sampled=n_sampled,
        num_classes=nVocab)

    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

'''
Here we're going to choose a few common words and few uncommon words. Then, we'll print out the closest words to them. 
It's a nice way to check that our embedding table is grouping together words with similar semantic meanings.
'''
with train_graph.as_default():
    validSize = 16
    validWindow = 100

    validExamples = np.array(random.sample(range(validWindow), validSize // 2))
    validExamples = np.append(validExamples,
                               random.sample(range(1000, 1000 + validWindow), validSize // 2))

    validDataset = tf.constant(validExamples, dtype=tf.int32)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalizedEmbedding = embedding / norm
    valid_embedding = tf.nn.embedding_lookup(normalizedEmbedding, validDataset)
    similarity = tf.matmul(valid_embedding, tf.transpose(normalizedEmbedding))


'''
Train the network. Every 100 batches it reports the training loss. Every 1000 batches, it'll print out the validation
words.
'''
epochs = 10
batch_size = 1000
window_size = 10

with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = getBatches(trainWords, batch_size, window_size)
        start = time.time()
        for x, y in batches:

            feed = {netInputs: x,
                    netLabels: np.array(y)[:, None]}
            trainLoss, _ = sess.run([cost, optimizer], feed_dict=feed)

            loss += trainLoss

            if iteration % 100 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss / 100),
                      "{:.4f} sec/batch".format((end - start) / 100))
                loss = 0
                start = time.time()

            if iteration % 1000 == 0:
                sim = similarity.eval()
                for i in range(validSize):
                    validWord = int2vocab[validExamples[i]]
                    topK = 8
                    nearest = (-sim[i, :]).argsort()[1:topK + 1]
                    log = 'Nearest to %s:' % validWord
                    for k in range(topK):
                        closeWord = int2vocab[nearest[k]]
                        logStatement = '%s %s,' % (log, closeWord)
                    print(logStatement)

            iteration += 1
    save_path = saver.save(sess, "checkpoints/text8.ckpt")
    embed_mat = sess.run(normalizedEmbedding)


'''
Restore the trained network if you need to
'''
with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    embed_mat = sess.run(embedding)


'''
Below we'll use T-SNE to visualize how our high-dimensional word vectors cluster together. T-SNE is used to project 
these vectors into two dimensions while preserving local structure. 
'''
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
vizWords = 500
tsne = TSNE()
embedTSNE = tsne.fit_transform(embed_mat[:vizWords, :])

fig, ax = plt.subplots(figsize=(14, 14))
for idx in range(vizWords):
    plt.scatter(*embedTSNE[idx, :], color='steelblue')
    plt.annotate(int2vocab[idx], (embedTSNE[idx, 0], embedTSNE[idx, 1]), alpha=0.7)
