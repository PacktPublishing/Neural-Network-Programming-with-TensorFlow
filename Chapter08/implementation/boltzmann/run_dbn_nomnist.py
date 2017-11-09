import tensorflow as tf
import numpy as np

import cPickle as pickle

from common.models.boltzmann import dbn
from common.utils import datasets, utilities


flags = tf.app.flags
FLAGS = flags.FLAGS
pickle_file = '../notMNIST.pickle'

image_size = 28
num_of_labels = 10

RELU = 'RELU'
RELU6 = 'RELU6'
CRELU = 'CRELU'
SIGMOID = 'SIGMOID'
ELU = 'ELU'
SOFTPLUS = 'SOFTPLUS'

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_of_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

    def activation(name, features):
        if name == RELU:
            return tf.nn.relu(features)
        if name == RELU6:
            return tf.nn.relu6(features)
        if name == SIGMOID:
            return tf.nn.sigmoid(features)
        if name == CRELU:
            return tf.nn.crelu(features)
        if name == ELU:
            return tf.nn.elu(features)
        if name == SOFTPLUS:
            return tf.nn.softplus(features)

if __name__ == '__main__':

    utilities.random_seed_np_tf(-1)

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

    #trainX, trainY, validX, validY, testX, testY = datasets.load_mnist_dataset(mode='supervised')
    trainX = train_dataset
    trainY = train_labels

    validX = valid_dataset
    validY = valid_labels
    testX = test_dataset
    testY = test_labels

    finetune_act_func = tf.nn.relu
    rbm_layers = [256]
    do_pretrain = True



    name = 'dbn'
    rbm_layers = [256]
    finetune_act_func ='relu'
    do_pretrain = True

    rbm_learning_rate = [0.001]

    rbm_num_epochs = [1]
    rbm_gibbs_k= [1]
    rbm_stddev= 0.1
    rbm_gauss_visible= False
    momentum= 0.5
    rbm_batch_size= [32]
    finetune_learning_rate = 0.01
    finetune_num_epochs = 1
    finetune_batch_size = 32
    finetune_opt = 'momentum'
    finetune_loss_func = 'softmax_cross_entropy'

    finetune_dropout = 1
    finetune_act_func = tf.nn.sigmoid


    srbm = dbn.DeepBeliefNetwork(
        name=name, do_pretrain=do_pretrain,
        rbm_layers=rbm_layers,
        finetune_act_func=finetune_act_func, rbm_learning_rate=rbm_learning_rate,
        rbm_num_epochs=rbm_num_epochs, rbm_gibbs_k = rbm_gibbs_k,
        rbm_gauss_visible=rbm_gauss_visible, rbm_stddev=rbm_stddev,
        momentum=momentum, rbm_batch_size=rbm_batch_size, finetune_learning_rate=finetune_learning_rate,
        finetune_num_epochs=finetune_num_epochs, finetune_batch_size=finetune_batch_size,
        finetune_opt=finetune_opt, finetune_loss_func=finetune_loss_func,
        finetune_dropout=finetune_dropout
        )

    print(do_pretrain)
    if do_pretrain:
        srbm.pretrain(trainX, validX)

    # finetuning
    print('Start deep belief net finetuning...')
    srbm.fit(trainX, trainY, validX, validY)

    # Test the model
    print('Test set accuracy: {}'.format(srbm.score(testX, testY)))
