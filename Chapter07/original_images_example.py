
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

class OriginalImages:

    def __init__(self):
        pass

    def main(self):
        X_train, X_test = self.standard_scale(mnist.train.images, mnist.test.images)

        original_imgs = X_test[:100]
        plt.figure(1, figsize=(10, 10))

        for i in range(0, 100):
            im = original_imgs[i].reshape((28, 28))
            ax = plt.subplot(10, 10, i + 1)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(8)

            plt.imshow(im, cmap="gray", clim=(0.0, 1.0))
        plt.suptitle(' Original Images', fontsize=15, y=0.95)
        plt.savefig('figures/original_images.png')
        plt.show()


def main():
    auto = OriginalImages()
    auto.main()

if __name__ == '__main__':
    main()
