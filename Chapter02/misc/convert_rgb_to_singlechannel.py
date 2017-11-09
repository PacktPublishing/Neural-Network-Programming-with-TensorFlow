import tensorflow as tf

home = '/home/ubuntu/work/github/rajdeepd/neuralnetwork-programming/ch02/'
path = home +  'data/train-200-200/Type_1/7.png'


rgb_image = tf.image.decode_png(path)
#rgb_image_float = tf.image.convert_image_dtype(rgb_image, tf.float32)
greyscale = tf.image.rgb_to_grayscale(rgb_image)
print(greyscale)

path2 = home + 'nomnist/notMNIST_small/A/' + 'SHVtYW5pc3QgNTMxIEJsYWNrIEJULnR0Zg==.png'

image2 = tf.image.decode_png(path2, channels=1)
image3 = tf.squeeze(image2)
print(image3)


pil_buf = open(path2).read()
contents = tf.placeholder(dtype=tf.string)
decode_op = tf.image.decode_png(contents, channels=1)
gray_image = tf.squeeze(decode_op) # shape (127,127,1) -> shape (127,127)
sess = tf.create_session()
[decoded] = sess.run([gray_image], feed_dict={contents: pil_buf})
print(decoded)