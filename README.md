


# Neural Network Programming with Tensorflow
This is the code repository for [Neural Network Programming with Tensorflow](https://www.packtpub.com/big-data-and-business-intelligence/neural-network-programming-tensorflow?utm_source=github&utm_medium=repository&utm_campaign=9781788390392), published by [Packt](https://www.packtpub.com/?utm_source=github). It contains all the supporting project files necessary to work through the book from start to finish.
## About the Book
You will start by getting a quick overview of the popular TensorFlow library and how it is used to train different neural networks. You will get a thorough understanding of the fundamentals and basic math for neural networks and why TensorFlow is a popular choice Then, you will proceed to implement a simple feed forward neural network. Next you will master optimization techniques and algorithms for neural networks using TensorFlow. Further, you will learn to implement some more complex types of neural networks such as convolutional neural networks, recurrent neural networks, and Deep Belief Networks. In the course of the book, you will be working on real-world datasets to get a hands-on understanding of neural network programming. You will also get to train generative models and will learn the applications of autoencoders.
## Instructions and Navigation
All of the code is organized into folders. Each folder starts with a number followed by the application name. For example, Chapter02.



The code will look like the following:
```
import tensorflow as tf
# equation 1
x1 = tf.constant(3, dtype=tf.float32)
y1 = tf.constant(2, dtype=tf.float32)
point1 = tf.stack([x1, y1])
# equation 2
x2 = tf.constant(4, dtype=tf.float32)
y2 = tf.constant(-1, dtype=tf.float32)
point2 = tf.stack([x2, y2])
# solve for AX=C
X = tf.transpose(tf.stack([point1, point2]))
C = tf.ones((1,2), dtype=tf.float32)
A = tf.matmul(C, tf.matrix_inverse(X))
with tf.Session() as sess:
X = sess.run(X)
print(X)
A = sess.run(A)
print(A)
b = 1 / A[0][1]
a = -b * A[0][0]
print("Hence Linear Equation is: y = {a}x + {b}".format(a=a, b=b))
```

This book will guide you through the installation of all the tools that you need to follow the
examples:
Python 3.4 or above
TensorFlow r.14 or above

## Related Products
* [Predictive Analytics with TensorFlow](https://www.packtpub.com/big-data-and-business-intelligence/predictive-analytics-tensorflow?utm_source=github&utm_medium=repository&utm_campaign=9781788398923)

* [Deep Learning with TensorFlow](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-tensorflow?utm_source=github&utm_medium=repository&utm_campaign=9781786469786)

* [Getting Started with TensorFlow](https://www.packtpub.com/big-data-and-business-intelligence/getting-started-tensorflow?utm_source=github&utm_medium=repository&utm_campaign=9781786468574)

### Download a free PDF

 <i>If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost.<br>Simply click on the link to claim your free PDF.</i>
<p align="center"> <a href="https://packt.link/free-ebook/9781788390392">https://packt.link/free-ebook/9781788390392 </a> </p>