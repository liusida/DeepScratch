import tensorflow as tf
import numpy as np

X = tf.constant(np.arange(12).reshape(3,4).astype(float))
Y = tf.constant(np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0]]).astype(float))
Y_hat = tf.nn.softmax(X)
L1 = tf.nn.softmax_cross_entropy_with_logits(logits=X, labels=Y)
L = tf.reduce_sum(L1)

Gradient = tf.gradients(L, X)
M = Y_hat - Y

with tf.Session() as s:
    x,y,y_hat,l,g,m = s.run([X, Y, Y_hat, L, Gradient, M], feed_dict={})
print("X ",x,"\n\nY ",y,"\n\nY_hat ",y_hat,"\n\nL ",l,"\n\nG ",g,"\n\nM ",m)