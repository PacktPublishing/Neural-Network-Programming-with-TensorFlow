import tensorflow as tf
image_batch = tf.constant([

        [  # First Image
                [[0, 255, 0], [0, 255, 0], [0, 255, 0]],
                [[0, 255, 0], [0, 255, 0], [0, 255, 0]]
        ],
        [  # Second Image
                [[0, 0, 255], [0, 0, 255], [0, 0, 255]],
                [[0, 0, 255], [0, 0, 255], [0, 0, 255]]
        ]
    ])
print(image_batch.get_shape())

session = tf.InteractiveSession()

print(session.eval(image_batch[0][0][0]))