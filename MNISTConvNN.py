# -*- coding: utf-8 -*-
"""
MNIST digit classification with a convolutional neural network
"""

#Convolutional neural network
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


#Weight function
def weight_var(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#Bias function
def bias_var(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

#2x2 blocks max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#Layer 1. 32 features for each 5x5 patch
W_conv1 = weight_var([5,5,1,32])
b_conv1 = bias_var([32])

#Reshape x into a 4d tensor (image size 28x28)
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Layer 2
W_conv2 = weight_var([5,5,32,64])
b_conv2 = bias_var([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_var([7*7*64,1024])
b_fc1 = bias_var([1024])

#Reducing the images to 7x7
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.placeholder(h_fc1, keep_prob)

#Readout layer
W_fc2 = weight_var([1024, 10])
b_fc2 = bias_var([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
corr_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(corr_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch = mnist.train_next_batch(50)
        if i % 100 == 0:
            train_acc = acc.eval(feed_dict={x: batch[0], y_:batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g ' % (1, train_acc))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) 
        
    print('test accuracy %g ' % acc.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}))