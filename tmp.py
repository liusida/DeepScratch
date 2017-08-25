import tensorflow as tf
import numpy as np

Y_hat = np.array([[1.0,2,3],[4,3,4]]).T
Y = np.array([[1.,0,0], [0,1,1]]).T
tf_softmax = tf.nn.softmax(logits=Y_hat)
tf_cost = -tf.reduce_sum(Y * tf.log(tf_softmax), axis=1)
tf_cost1 = tf.nn.softmax_cross_entropy_with_logits(logits=Y_hat, labels=Y)
with tf.Session() as s:
    softmax, cost, cost1 = s.run([tf_softmax, tf_cost, tf_cost1])
softmax = softmax.T
print("softmax = ", softmax)
print("cost = ", cost)
print("cost1 = ", cost1)

