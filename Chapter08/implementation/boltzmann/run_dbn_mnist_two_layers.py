import tensorflow as tf

from common.models.boltzmann import dbn
from common.utils import datasets, utilities


flags = tf.app.flags
FLAGS = flags.FLAGS

if __name__ == '__main__':

    utilities.random_seed_np_tf(-1)

    trainX, trainY, validX, validY, testX, testY = datasets.load_mnist_dataset(mode='supervised')

    finetune_act_func = tf.nn.relu
    do_pretrain = True



    name = 'dbn'
    rbm_layers = [256, 256]
    finetune_act_func ='relu'
    do_pretrain = True

    rbm_learning_rate = [0.001, 0001]

    rbm_num_epochs = [5, 5]
    rbm_gibbs_k= [1, 1]
    rbm_stddev= 0.1
    rbm_gauss_visible= False
    momentum= 0.5
    rbm_batch_size= [32, 32]
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
