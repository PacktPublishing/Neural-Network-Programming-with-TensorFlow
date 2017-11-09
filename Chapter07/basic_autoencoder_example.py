import numpy as np

import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from autencoder_models.auto_encoder import AutoEncoder
import math
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

class BasicAutoEncoder:

    def __init__(self):
        pass

    def standard_scale(self,X_train, X_test):
        preprocessor = prep.StandardScaler().fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)
        return X_train, X_test

    def get_random_block_from_data(self,data, batch_size):
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]

    def main(self):
        X_train, X_test = self.standard_scale(mnist.train.images, mnist.test.images)

        n_samples = int(mnist.train.num_examples)
        training_epochs = 5
        batch_size = 128
        display_step = 1

        autoencoder = AutoEncoder(n_input = 784,
                                  n_hidden = 200,
                                  transfer_function = tf.nn.softplus,
                                  optimizer = tf.train.AdamOptimizer(learning_rate = 0.001))

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = self.get_random_block_from_data(X_train, batch_size)

                # Fit training using batch data
                cost = autoencoder.partial_fit(batch_xs)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))

        wts = autoencoder.getWeights()
        dim = math.ceil(math.sqrt(autoencoder.n_hidden))
        plt.figure(1, figsize=(dim, dim))

        for i in range(0, autoencoder.n_hidden):
            im = wts.flatten()[i::autoencoder.n_hidden].reshape((28, 28))
            ax = plt.subplot(dim, dim, i + 1)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontname('Arial')
                label.set_fontsize(8)
            # plt.title('Feature Weights ' + str(i))
            plt.imshow(im, cmap="gray", clim=(-1.0, 1.0))
        plt.suptitle('Basic AutoEncoder Weights', fontsize=15, y=0.95)
        #plt.title("Test Title", y=1.05)
        plt.savefig('figures/basic_autoencoder_weights.png')
        plt.show()

        predicted_imgs = autoencoder.reconstruct(X_test[:100])

        plt.figure(1, figsize=(10, 10))

        for i in range(0, 100):
            im = predicted_imgs[i].reshape((28, 28))
            ax = plt.subplot(10, 10, i + 1)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                        label.set_fontname('Arial')
                        label.set_fontsize(8)

            plt.imshow(im, cmap="gray", clim=(0.0, 1.0))
        plt.suptitle('Basic AutoEncoder Images', fontsize=15, y=0.95)
        plt.savefig('figures/basic_autoencoder_images.png')
        plt.show()

        original_imgs = X_test[:100]
        plt.figure(1, figsize=(10, 10))

        for i in range(0, 100):
            im = original_imgs[i].reshape((28, 28))
            ax = plt.subplot(10, 10, i + 1)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                #label.set_fontname('Arial')
                label.set_fontsize(8)

            plt.imshow(im, cmap="gray", clim=(0.0, 1.0))
        plt.suptitle(' Original Images', fontsize=15, y=0.95)
        plt.savefig('figures/original_images.png')
        plt.show()


def main():
    auto = BasicAutoEncoder()
    auto.main()

if __name__ == '__main__':
    main()
