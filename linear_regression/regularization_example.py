import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from libs.data_utils import split_dataset

"""
Usage:
    python regularization_example.py
"""

learning_rate = 0.01
training_epochs = 100
reg_lambda = 0.0

x_dataset = np.linspace(-1, 1, 100)

num_coeffs = 9
y_dataset_params = [0.] * num_coeffs
y_dataset_params[2] = 1
y_dataset = 0

for i in range(num_coeffs):
    y_dataset += y_dataset_params[i] * np.power(x_dataset, i)

y_dataset += np.random.randn(*x_dataset.shape) * 0.3

(x_train, x_test, y_train, y_test) = split_dataset(x_dataset, y_dataset, 0.7)

X = tf.placeholder("float")
Y = tf.placeholder("float")


def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)

w = tf.Variable([0.] * num_coeffs, name="weights")
y_model = model(X, w)
cost = tf.divide(tf.add(tf.reduce_sum(tf.square(Y - y_model)),
                        tf.multiply(reg_lambda, tf.reduce_sum(tf.square(w)))),
                 2 * x_train.size)

train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

for reg_lambda in np.linspace(0, 0., 100):
    for epoch in range(training_epochs):
        sess.run(train_op, feed_dict={
            X: x_train,
            Y: y_train
        })
    final_cost = sess.run(cost, feed_dict={
        X: x_test,
        Y: y_test
    })
    print("reg lambda", reg_lambda)
    print("final cost", final_cost)


w_val = sess.run(w)
print("w_val = ", w_val)

sess.close()

plt.scatter(x_dataset, y_dataset)
y_pred = 0
for i in range(num_coeffs):
    y_pred += w_val[i] * np.power(x_dataset, i)

plt.plot(x_dataset, y_pred, 'r')
plt.show()