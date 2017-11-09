import tensorflow as tf


def main():
    session = tf.InteractiveSession()
    input_batch = tf.constant([
        [  # First Input (6x6x1)
            [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
            [[0.1], [1.1], [2.1], [3.1], [4.1], [5.1]],
            [[0.2], [1.2], [2.2], [3.2], [4.2], [5.2]],
            [[0.3], [1.3], [2.3], [3.3], [4.3], [5.3]],
            [[0.4], [1.4], [2.4], [3.4], [4.4], [5.4]],
            [[0.5], [1.5], [2.5], [3.5], [4.5], [5.5]],
        ],
    ])

    kernel = tf.constant([  # Kernel (3x3x1)
        [[[0.0]], [[0.5]], [[0.0]]],
        [[[0.0]], [[0.5]], [[0.0]]],
        [[[0.0]], [[0.5]], [[0.0]]]
    ])

    # NOTE: the change in the size of the strides parameter.
    conv2d = tf.nn.conv2d(input_batch, kernel, strides=[1, 3, 3, 1], padding='SAME')
    conv2d_output = session.run(conv2d)
    print(conv2d_output)

    conv2d = tf.nn.conv2d(input_batch, kernel, strides=[1, 3, 3, 1], padding='VALID')
    conv2d_output = session.run(conv2d)
    print(conv2d_output)

if __name__ == '__main__':
  main()