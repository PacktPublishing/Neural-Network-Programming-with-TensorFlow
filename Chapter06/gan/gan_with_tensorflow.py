import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer

sns.set(color_codes=True)

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


# gaussian data distribution
class DataDist(object):
    def __init__(self):
        self.mue = 4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mue, self.sigma, N)
        samples.sort()
        return samples


# data distribution with noise
class GeneratorDist(object):
    def __init__(self, rnge):
        self.rnge = rnge

    def sample(self, N):
        return np.linspace(-self.rnge, self.rnge, N) + \
               np.random.random(N) * 0.01


# linear method
def linearUnit(input, output_dim, scope=None, stddev=1.0):
    with tf.variable_scope(scope or 'linearUnit'):
        weight = tf.get_variable(
            'weight',
            [input.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        bias = tf.get_variable(
            'bias',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input, weight) + bias


# generator network
def generatorNetwork(input, hidden_size):
    hidd0 = tf.nn.softplus(linearUnit(input, hidden_size, 'g0'))
    hidd1 = linearUnit(hidd0, 1, 'g1')
    return hidd1


# discriminator network
def discriminatorNetwork(input, h_dim, minibatch_layer=True):
    hidd0 = tf.nn.relu(linearUnit(input, h_dim * 2, 'd0'))
    hidd1 = tf.nn.relu(linearUnit(hidd0, h_dim * 2, 'd1'))

    if minibatch_layer:
        hidd2 = miniBatch(hidd1)
    else:
        hidd2 = tf.nn.relu(linearUnit(hidd1, h_dim * 2, scope='d2'))

    hidd3 = tf.sigmoid(linearUnit(hidd2, 1, scope='d3'))
    return hidd3


# minibatch
def miniBatch(input, numKernels=5, kernelDim=3):
    x = linearUnit(input, numKernels * kernelDim, scope='minibatch', stddev=0.02)
    act = tf.reshape(x, (-1, numKernels, kernelDim))
    differences = tf.expand_dims(act, 3) - \
            tf.expand_dims(tf.transpose(act, [1, 2, 0]), 0)
    absDiffs = tf.reduce_sum(tf.abs(differences), 2)
    minibatchFeatures = tf.reduce_sum(tf.exp(-absDiffs), 2)
    return tf.concat([input, minibatchFeatures], 1)


# optimizer
def optimizer(loss, var_list):
    learning_rate = 0.001
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=step,
        var_list=var_list
    )
    return optimizer


# log
def log(x):
    return tf.log(tf.maximum(x, 1e-5))


class GAN(object):
    def __init__(self, params):
        with tf.variable_scope('Generator'):
            self.zee = tf.placeholder(tf.float32, shape=(params.batchSize, 1))
            self.Gee = generatorNetwork(self.zee, params.hidden_size)

        self.xVal = tf.placeholder(tf.float32, shape=(params.batchSize, 1))
        with tf.variable_scope('Discriminator'):
            self.Dis1 = discriminatorNetwork(
                self.xVal,
                params.hidden_size,
                params.minibatch
            )
        with tf.variable_scope('D', reuse=True):
            self.Dis2 = discriminatorNetwork(
                self.Gee,
                params.hidden_size,
                params.minibatch
            )

        self.lossD = tf.reduce_mean(-log(self.Dis1) - log(1 - self.Dis2))
        self.lossG = tf.reduce_mean(-log(self.Dis2))

        vars = tf.trainable_variables()
        self.dParams = [v for v in vars if v.name.startswith('D/')]
        self.gParams = [v for v in vars if v.name.startswith('G/')]

        self.optD = optimizer(self.lossD, self.dParams)
        self.optG = optimizer(self.lossG, self.gParams)


'''
Train GAN model
'''
def trainGan(model, data, gen, params):
    animFrames = []

    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        for step in range(params.numSteps + 1):
            x = data.sample(params.batchSize)
            z = gen.sample(params.batchSize)
            lossD, _, = session.run([model.lossD, model.optD], {
                model.x: np.reshape(x, (params.batchSize, 1)),
                model.z: np.reshape(z, (params.batchSize, 1))
            })

            z = gen.sample(params.batchSize)
            lossG, _ = session.run([model.lossG, model.optG], {
                model.z: np.reshape(z, (params.batchSize, 1))
            })

            if step % params.log_every == 0:
                print('{}: {:.4f}\t{:.4f}'.format(step, lossD, lossG))

            if params.animPath and (step % params.animEvery == 0):
                animFrames.append(
                    getSamples(model, session, data, gen.range, params.batchSize)
                )

        if params.animPath:
            saveAnimation(animFrames, params.animPath, gen.range)
        else:
            samps = getSamples(model, session, data, gen.range, params.batchSize)
            plotDistributions(samps, gen.range)


def getSamples(
        model,
        session,
        data,
        sampleRange,
        batchSize,
        numPoints=10000,
        numBins=100
):
    xs = np.linspace(-sampleRange, sampleRange, numPoints)
    binss = np.linspace(-sampleRange, sampleRange, numBins)

    # decision boundary
    db = np.zeros((numPoints, 1))
    for i in range(numPoints // batchSize):
        db[batchSize * i:batchSize * (i + 1)] = session.run(
            model.D1,
            {
                model.x: np.reshape(
                    xs[batchSize * i:batchSize * (i + 1)],
                    (batchSize, 1)
                )
            }
        )

    # data distribution
    d = data.sample(numPoints)
    pds, _ = np.histogram(d, bins=binss, density=True)

    zs = np.linspace(-sampleRange, sampleRange, numPoints)
    g = np.zeros((numPoints, 1))
    for i in range(numPoints // batchSize):
        g[batchSize * i:batchSize * (i + 1)] = session.run(
            model.G,
            {
                model.z: np.reshape(
                    zs[batchSize * i:batchSize * (i + 1)],
                    (batchSize, 1)
                )
            }
        )
    pgs, _ = np.histogram(g, bins=binss, density=True)

    return db, pds, pgs


def plotDistributions(samps, sampleRange):
    db, pd, pg = samps
    dbX = np.linspace(-sampleRange, sampleRange, len(db))
    pX = np.linspace(-sampleRange, sampleRange, len(pd))
    f, ax = plt.subplots(1)
    ax.plot(dbX, db, label='Decision Boundary')
    ax.set_ylim(0, 1)
    plt.plot(pX, pd, label='Real Data')
    plt.plot(pX, pg, label='Generated Data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data Values')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()


def saveAnimation(animFrames, animPath, sampleRange):
    f, ax = plt.subplots(figsize=(6, 4))
    f.suptitle('1D GAN', fontsize=15)
    plt.xlabel('dataValues')
    plt.ylabel('probabilityDensity')
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1.4)
    lineDb, = ax.plot([], [], label='decision boundary')
    linePd, = ax.plot([], [], label='real data')
    linePg, = ax.plot([], [], label='generated data')
    frameNumber = ax.text(
        0.02,
        0.95,
        '',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes
    )
    ax.legend()

    db, pd, _ = animFrames[0]
    dbX = np.linspace(-sampleRange, sampleRange, len(db))
    pX = np.linspace(-sampleRange, sampleRange, len(pd))

    def init():
        lineDb.set_data([], [])
        linePd.set_data([], [])
        linePg.set_data([], [])
        frameNumber.set_text('')
        return (lineDb, linePd, linePg, frameNumber)

    def animate(i):
        frameNumber.set_text(
            'Frame: {}/{}'.format(i, len(animFrames))
        )
        db, pd, pg = animFrames[i]
        lineDb.set_data(dbX, db)
        linePd.set_data(pX, pd)
        linePg.set_data(pX, pg)
        return (lineDb, linePd, linePg, frameNumber)

    anim = animation.FuncAnimation(
        f,
        animate,
        init_func=init,
        frames=len(animFrames),
        blit=True
    )
    anim.save(animPath, fps=30, extra_args=['-vcodec', 'libx264'])


# start gan modeling
def gan(args):
    model = GAN(args)
    trainGan(model, DataDist(), GeneratorDist(range=8), args)


# input arguments
def parseArguments():
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--num-steps', type=int, default=5000,
                           help='the number of training steps to take')
    argParser.add_argument('--hidden-size', type=int, default=4,
                           help='MLP hidden size')
    argParser.add_argument('--batch-size', type=int, default=8,
                           help='the batch size')
    argParser.add_argument('--minibatch', action='store_true',
                           help='use minibatch discrimination')
    argParser.add_argument('--log-every', type=int, default=10,
                           help='print loss after this many steps')
    argParser.add_argument('--anim-path', type=str, default=None,
                           help='path to the output animation file')
    argParser.add_argument('--anim-every', type=int, default=1,
                           help='save every Nth frame for animation')
    return argParser.parse_args()


# start the gan app
if __name__ == '__main__':
    gan(parseArguments())
