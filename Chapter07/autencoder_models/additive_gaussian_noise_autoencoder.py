
import numpy as np
import tensorflow as tf

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)

class AdditiveGaussianNoiseAutoEncoder(object):

    def __init__(self, num_input, num_hidden, transfer_function=tf.nn.sigmoid, optimizer=tf.train.AdamOptimizer(),
                 scale=0.1):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        n_weights = self._initialize_weights()
        self.weights = n_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.num_input])

        self.hidden_layer = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((num_input,)),
                                                           self.weights['w1']),
                                                 self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden_layer, self.weights['w2']), self.weights['b2'])

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))

        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def _initialize_weights(self):
        weights = dict()
        weights['w1'] = tf.Variable(xavier_init(self.num_input, self.num_hidden))
        weights['b1'] = tf.Variable(tf.zeros([self.num_hidden], dtype=tf.float32))
        weights['w2'] = tf.Variable(tf.zeros([self.num_hidden, self.num_input], dtype=tf.float32))
        weights['b2'] = tf.Variable(tf.zeros([self.num_input], dtype=tf.float32))
        return weights

    def partial_fit(self, X):
        cost, opt = self.session.run((self.cost, self.optimizer), feed_dict={self.x: X,
                                                                             self.scale: self.training_scale
                                                                             })
        return cost

    def kl_divergence(self, p, p_hat):
        return tf.reduce_mean(p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat))

    def calculate_total_cost(self, X):
        return self.session.run(self.cost, feed_dict={self.x: X,
                                                      self.scale: self.training_scale
                                                      })

    def transform(self, X):
        return self.session.run(self.hidden_layer, feed_dict={self.x: X,
                                                              self.scale: self.training_scale
                                                              })

    def generate_value(self, _hidden=None):
        if _hidden is None:
            _hidden = np.random.normal(size=self.weights["b1"])
        return self.session.run(self.reconstruction, feed_dict={self.hidden_layer: _hidden})

    def reconstruct(self, X):
        return self.session.run(self.reconstruction, feed_dict={self.x: X,
                                                                self.scale: self.training_scale
                                                                })

    def get_weights(self):
        return self.session.run(self.weights['w1'])


    def get_biases(self):
        return self.session.run(self.weights['b1'])
