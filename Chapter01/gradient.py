# python gradient.py --logdir=/neuralnetwork-programming/ch01/logs
# tensorboard --logdir=/neuralnetwork-programming/ch01/logs
# http://localhost:6006/

import tensorflow as tf

#function to transform gradients
def T(g, decay=1.0):
    #return decayed gradient
    return decay*g

# x variable
x = tf.Variable(10.0,name='x')
# b placeholder (simulates the "data" part of the training)
b = tf.placeholder(tf.float32)
# make model (1/2)(x-b)^2
xx_b = 0.5*tf.pow(x-b,2)
y=xx_b

learning_rate = 1.0
opt = tf.train.GradientDescentOptimizer(learning_rate)
# gradient variable list = [ (gradient,variable) ]
gv = opt.compute_gradients(y,[x])
# transformed gradient variable list = [ (T(gradient),variable) ]
decay = 0.9 # decay the gradient for the sake of the example
tgv = [ (T(g,decay=decay), v) for (g,v) in gv] #list [(grad,var)]
# apply transformed gradients (this case no transform)
apply_transform_op = opt.apply_gradients(tgv)

(dydx,_) = tgv[0]
x_scalar_summary = tf.summary.scalar("x", x)
grad_scalar_summary = tf.summary.scalar("dydx", dydx)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    tensorboard_data_dump = '/Users/manpreet.singh/Sandbox/codehub/github/science/neuralnetwork-programming/ch01/logs'
    writer = tf.summary.FileWriter(tensorboard_data_dump, sess.graph)

    sess.run(tf.initialize_all_variables())
    epochs = 14
    for i in range(epochs):
        b_val = 1.0
        print('----')
        x_before_update = x.eval()
        print('before update',x_before_update)

        # get gradients
        #grad_list = [g for (g,v) in gv]
        (summary_str_grad,grad_val) = sess.run([merged] + [dydx], feed_dict={b: b_val})
        grad_vals = sess.run([g for (g,v) in gv], feed_dict={b: b_val})
        print('grad_vals: ',grad_vals)
        writer.add_summary(summary_str_grad, i)

        # applies the gradients
        [summary_str_apply_transform,_] = sess.run([merged,apply_transform_op], feed_dict={b: b_val})
        writer.add_summary(summary_str_apply_transform, i)

        print('value of x after update should be: ', x_before_update - T(grad_vals[0], decay=decay))
        x_after_update = x.eval()
        print('after update', x_after_update)