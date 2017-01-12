import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Usage:
    python linear_regression.py
"""

learning_rate = 0.01
training_epochs = 100

x_train = np.linspace(-1, 1, 200)
y_train = 2 * x_train + np.random.rand(*x_train.shape) * 0.33

X = tf.placeholder("float")
Y = tf.placeholder("float")


def model(X, w):
    return tf.multiply(X, w)

w = tf.Variable(0.0, name="weights")

y_model = model(X, w)
cost = (tf.square(Y - y_model))

train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in tqdm(range(training_epochs)):
    for (x, y) in zip(x_train, y_train):
        sess.run(train_op, feed_dict={
            X: x,
            Y: y
        })
    w_val = sess.run(w)

sess.close()

print ("w_val = %f" % w_val)

plt.scatter(x_train, y_train)
y_learned = x_train * w_val
plt.plot(x_train, y_learned, 'r')
plt.show()