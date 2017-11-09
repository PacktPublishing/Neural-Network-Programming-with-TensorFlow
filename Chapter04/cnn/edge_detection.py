import matplotlib as mil
import tensorflow as tf

from matplotlib import pyplot
fig = pyplot.gcf()
fig.set_size_inches(4, 4)

sess = tf.InteractiveSession()

image_filename = "/home/ubuntu/Downloads/n02107142_16917.jpg"


filename_queue = tf.train.string_input_producer([image_filename]) #  list of files to read

reader = tf.WholeFileReader()
try:
  image_reader = tf.WholeFileReader()
  _, image_file = image_reader.read(filename_queue)
  image = tf.image.decode_jpeg(image_file)
  print(image)

except Exception as e:
    print(e)


sess.run(tf.initialize_all_variables())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

image_batch = tf.image.convert_image_dtype(tf.expand_dims(image, 0), tf.float32, saturate=False)


# In[8]:


kernel = tf.constant([
        [
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]]
        ],
        [
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
            [[ 8., 0., 0.], [ 0., 8., 0.], [ 0., 0., 8.]],
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]]
        ],
        [
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]]
        ]
    ])


conv2d = tf.nn.conv2d(image_batch, kernel, [1, 1, 1, 1], padding="SAME")
activation_map = sess.run(tf.minimum(tf.nn.relu(conv2d), 255))
fig = pyplot.gcf()
pyplot.imshow(activation_map[0], interpolation='nearest')
fig.set_size_inches(4, 4)
fig.savefig("./example-edge-detection.png")
#pyplot.show()