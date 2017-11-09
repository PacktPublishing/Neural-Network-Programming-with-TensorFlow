import tensorflow as tf


class AutoEncoder:

    def __init__(self, num_input, num_hidden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer()):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.transfer = transfer_function

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model for reconstruction of the image
        self.x_var = tf.placeholder(tf.float32, [None, self.num_input])
        self.hidden_layer = self.transfer(tf.add(tf.matmul(self.x_var, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden_layer, self.weights['w2']), self.weights['b2'])

        # cost function
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x_var), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        initializer = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(initializer)

    def _initialize_weights(self):
        weights = dict()
        weights['w1'] = tf.get_variable("w1", shape=[self.num_input, self.num_hidden],
                          initializer=tf.contrib.layers.xavier_initializer())
        weights['b1'] = tf.Variable(tf.zeros([self.num_hidden], dtype=tf.float32))
        weights['w2'] = tf.Variable(tf.zeros([self.num_hidden, self.num_input], dtype=tf.float32))
        weights['b2'] = tf.Variable(tf.zeros([self.num_input], dtype=tf.float32))
        return weights

    def partial_fit(self, X):
        cost, opt = self.session.run((self.cost, self.optimizer), feed_dict={self.x_var: X})
        return cost

    def calculate_total_cost(self, X):
        return self.session.run(self.cost, feed_dict = {self.x_var: X})

    def transform(self, X):
        return self.session.run(self.hidden_layer, feed_dict={self.x_var: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = self.session.run(tf.random_normal([1, self.num_hidden]))
        return self.session.run(self.reconstruction, feed_dict={self.hidden_layer: hidden})

    def reconstruct(self, X):
        return self.session.run(self.reconstruction, feed_dict={self.x_var: X})

    def get_weights(self):
        return self.session.run(self.weights['w1'])

    def get_biases(self):
        return self.session.run(self.weights['b1'])
