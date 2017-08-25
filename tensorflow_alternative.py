"""
Because I have a bug in numpy implemention, and I can not 
address it.
"""

import tensorflow as tf
from test import *

def ReLU(X):
    tf_X = tf.constant(X, dtype=tf.float64)
    tf_Y = tf.nn.relu(tf_X, name='relu')
    with tf.Session() as s:
        Y = s.run(tf_Y)
    return Y

def softmax(X):
    tf_X = tf.constant(X.T, dtype=tf.float64)
    tf_Y = tf.nn.softmax(tf_X, name='softmax')
    with tf.Session() as s:
        Y = s.run(tf_Y)
    Y = Y.T
    return Y

def forward_propagation_each_layer(W, A_prev, b, activation_function=ReLU):
    tf_W = tf.constant(W.T, dtype=tf.float64)
    tf_A_prev = tf.constant(A_prev.T, dtype=tf.float64)
    tf_b = tf.constant(b.reshape(-1), dtype=tf.float64)
    tf_Z = tf.nn.xw_plus_b(x=tf_A_prev, weights=tf_W, biases=tf_b)
    with tf.Session() as s:
        Z = s.run(tf_Z)
    Z = Z.T
    A = activation_function(Z)
    return Z, A

def loss(Y_hat, Y):
    m = Y_hat.shape[1]
    tf_Y_hat = tf.constant(Y_hat.T, name='Y_hat', dtype=tf.float64)
    tf_Y = tf.constant(Y.T, name='Y', dtype=tf.float64)
    tf_cost = -tf.reduce_sum(tf_Y * tf.log(tf_Y_hat), keep_dims=False)
    with tf.Session() as s:
        cost = s.run(tf_cost)
    cost = cost / m
    return cost

def backpropagate_cost(Y, AL):
    Y_hat = AL
    m = Y_hat.shape[1]
    tf_Y_hat = tf.constant(Y_hat.T, name='Y_hat', dtype=tf.float64)
    tf_Y = tf.constant(Y.T, name='Y', dtype=tf.float64)
    tf_cost = -tf.reduce_sum(tf_Y * tf.log(tf_Y_hat), keep_dims=False)
    tf_dAL = tf.gradients(tf_cost, tf_Y_hat)
    with tf.Session() as s:
        dAL = s.run(tf_dAL)
    dAL = np.array(dAL).T.reshape(AL.shape)
    return dAL

def backpropagate_softmax(AL, dAL, Y, ZL):
    tf_ZL = tf.constant(ZL.T, dtype=tf.float64)
    tf_Y = tf.constant(Y.T, name='Y', dtype=tf.float64)
    tf_cost = tf.nn.softmax_cross_entropy_with_logits(logits=tf_ZL, labels=tf_Y)
    tf_cost_scalar = tf.reduce_mean(tf_cost)
    tf_dZL = tf.gradients(tf_cost_scalar, tf_ZL)
    with tf.Session() as s:
        dZL, cost = s.run([tf_dZL, tf_cost])
    print("softmax_cross_entropy_with_logits = ", cost)
    dZL = np.array(dZL).T.reshape(ZL.shape)
    return dZL

test_ReLU(ReLU)
test_softmax(softmax)
test_forward_propagation_each_layer(forward_propagation_each_layer)
test_loss(loss, softmax)
test_backpropagation_cost(backpropagate_cost)
test_backpropagation_softmax(backpropagate_softmax, softmax, loss, backpropagate_cost)