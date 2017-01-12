import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Usage:
    python polynomial_model.py
"""
learning_rate = 0.01
training_epochs = 40

x_train = np.linspace(-1, 1, 101)

num_coeffs = 6
y_coeffs = [1, 2, 3, 4, 5, 6]
y_train = 0

for i in range(num_coeffs):
    y_train += y_coeffs[i] * np.power(x_train, i)

y_train += np.random.randn(*x_train.shape) * 1.5

X = tf.placeholder("float")
Y = tf.placeholder("float")


def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)

w = tf.Variable([0.] * num_coeffs, name="parameters")
y_model = model(X, w)

cost = (tf.pow(Y - y_model, 2))
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
print("w_val = ", w_val)

sess.close()

plt.scatter(x_train, y_train)
y_pred = 0
for i in range(num_coeffs):
    y_pred += w_val[i] * np.power(x_train, i)

plt.plot(x_train, y_pred, 'r')
plt.show()