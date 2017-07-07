# -*- coding: utf-8 -*-
"""
MNIST basic digit recognition
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot =True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

#placeholder for cross-entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))

"""
cross_entropy = tf.nn.(sparse_).softmax_with_logits
"""

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict ={x: batch_xs, y_: batch_ys})
    
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accu = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print(sess.run(accu, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))