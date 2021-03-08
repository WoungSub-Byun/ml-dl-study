# cost function in pure python

import numpy as np
import tensorflow as tf

X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])


def cost_func(W, X, Y):
    c = 0
    for i in range(len(X)):
        c += (W * X[i] - Y[i]) ** 2
    return c / len(X)


for feed_W in np.linspace(-3, 5, num=15):
    curr_cost = cost_func(feed_W, X, Y)
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))

# in tensorflow


def cost_func_tensorflow(W, X, Y):
    hypothesis = X * W
    return tf.reduce_mean(tf.square(hypothesis - Y))


W_values = np.linspace(-3, 5, num=15)
cost_values = []

for feed_W in np.linspace(-3, 5, num=15):
    curr_cost = cost_func_tensorflow(feed_W, X, Y)
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))

# Gradient Descent in tnesorflow

alpha = 0.01  # learning rate
gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))
descent = W - tf.multiply(alpha, gradient)
W.assign(descent)


# Gradient Descent 적용
tf.set_random_seed(0)

x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [1.0, 3.0, 5.0, 7.0]

W = tf.Variable(tf.random_normal([1], -100.0, 100.0))

for step in range(300):
    hypothesis = W * X  # simplified
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    alpha = 0.01  # learning rate
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))
    descent = W - tf.multiply(alpha, gradient)
    W.assign(descent)

    if step % 10 == 0:
        print("{:5} | {:10.4f} | {:10.6f}".format(step, cost.numpy(), W.numpy()[0]))
