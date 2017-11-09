import tensorflow as tf
import numpy as np


def main():
    # setup-only-ignore
    sess = tf.InteractiveSession()

    image_batch = tf.constant([
            [  # First Image
                [[255, 0, 0], [255, 0, 0], [0, 255, 0]],
                [[255, 0, 0], [255, 0, 0], [0, 255, 0]]
            ],
            [  # First Image
                [[0, 0, 0], [0, 255, 0], [0, 255, 0]],
                [[0, 0, 0], [0, 255, 0], [0, 255, 0]]
            ],
            [  # Second Image
                [[0, 0, 255], [0, 0, 255], [0, 0, 255]],
                [[0, 0, 255], [0, 0, 255], [0, 0, 255]]
            ]
        ])
    print(image_batch.get_shape())
    print(sess.run(image_batch)[0][0][0])

    i = tf.constant([
             [
                 [1.0, 1.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 1.0, 1.0],
                 [0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0]

             ]
        ])



    k = tf.constant([
            [
                [[1.0, 0.0, 1.0]],
                [[0.0, 1.0, 0.0]],
                [[1.0, 0.0, 1.0]]
            ]
        ])

    kernel = tf.reshape(k, [3, 3, 1, 1], name='kernel')
    image = tf.reshape(i, [1, 4, 5, 1], name='image')

    conv2d = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='VALID')
    res = tf.squeeze(tf.nn.conv2d(image, kernel, [1, 1, 1, 1], "VALID"))
    # VALID means no padding
    with tf.Session() as sess:
        print sess.run(res)
    #tensor_out = sess.run(conv2d)
    #print(tensor_out)
    lower_right_image_pixel = sess.run(input_batch)[0][1][1]
    lower_right_kernel_pixel = sess.run(conv2d)[0][1][1]

if __name__ == '__main__':
  main()