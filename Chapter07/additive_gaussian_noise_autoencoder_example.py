
import numpy as np
import random
import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from autencoder_models.additive_gaussian_noise_autoencoder import AdditiveGaussianNoiseAutoEncoder



class AdditiveGaussianNoiseAutoEncoderExample:

    def main(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        def get_random_block_from_data(data, batch_size):
            start_index = np.random.randint(0, len(data) - batch_size)
            return data[start_index:(start_index + batch_size)]

        X_train = mnist.train.images
        X_test = mnist.test.images

        n_samples = int(mnist.train.num_examples)
        training_epochs = 5
        batch_size = 128
        display_step = 1

        autoencoder = AdditiveGaussianNoiseAutoEncoder(num_input=784,
                                                       num_hidden=200,
                                                       transfer_function=tf.nn.sigmoid,
                                                       optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                                       scale=0.01)

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = get_random_block_from_data(X_train, batch_size)

                # Fit training using batch data
                cost = autoencoder.partial_fit(batch_xs)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", avg_cost)

        print("Total cost: " + str(autoencoder.calculate_total_cost(X_test)))

        # input weights
        wts = autoencoder.get_weights()
        dim = math.ceil(math.sqrt(autoencoder.num_hidden))
        plt.figure(1, figsize=(dim, dim))
        for i in range(0, autoencoder.num_hidden):
            im = wts.flatten()[i::autoencoder.num_hidden].reshape((28, 28))
            ax = plt.subplot(dim, dim, i + 1)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(8)
            #plt.title('Feature Weights ' + str(i))

            plt.imshow(im, cmap="gray", clim=(-1.0, 1.0))
        plt.suptitle('Additive Gaussian Noise AutoEncoder Weights', fontsize=15, y=0.95)
        plt.savefig('figures/additive_gaussian_weights.png')
        plt.show()


        predicted_imgs = autoencoder.reconstruct(X_test[:100])

        # plot the reconstructed images
        plt.figure(1, figsize=(10, 10))
        plt.title('Autoencoded Images')
        for i in range(0, 100):
            im = predicted_imgs[i].reshape((28, 28))
            ax = plt.subplot(10, 10, i + 1)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontname('Arial')
                label.set_fontsize(8)

            plt.imshow(im, cmap="gray", clim=(0.0, 1.0))
        plt.suptitle('Additive Gaussian Noise AutoEncoder Images', fontsize=15, y=0.95)
        plt.savefig('figures/additive_gaussian_images.png')
        plt.show()


def main():
    auto = AdditiveGaussianNoiseAutoEncoderExample()
    auto.main()

if __name__ == '__main__':
    main()
