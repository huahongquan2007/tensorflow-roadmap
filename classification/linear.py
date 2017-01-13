import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Usage:
    python linear.py
"""

x_label0 = np.random.normal(5, 1, 10)
x_label1 = np.random.normal(2, 1, 10)
xs = np.append(x_label0, x_label1)

labels = [0.] * len(x_label0) + [1.] * len(x_label1)

plt.scatter(xs, labels)


learning_rate = 0.001
training_epochs = 1000

X = tf.placeholder("float")
Y = tf.placeholder("float")


def model(X, w):
    return tf.add(tf.multiply(w[1], tf.pow(X, 1)),
                  tf.multiply(w[0], tf.pow(X, 0)))
    # y = w1 * x + w0

w = tf.Variable([0., 0.], name="weights")
y_model = model(X, w)
cost = tf.reduce_sum(tf.square(Y-y_model))

train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


for epoch in tqdm(range(training_epochs)):
    sess.run(train_op, feed_dict={
        X: xs,
        Y: labels
    })
    current_cost = sess.run(cost, feed_dict={
        X: xs,
        Y: labels
    })
    if epoch % 10 == 0:
        print(epoch, current_cost)

w_val = sess.run(w)
print("weights", w_val)

correct_prediction = tf.equal(Y, tf.to_float(tf.greater(y_model, 0.5)))
accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

print("accuracy", sess.run(accuracy, feed_dict={
    X: xs,
    Y: labels
}))

sess.close()

all_xs = np.linspace(0, 10, 100)
plt.plot(all_xs, all_xs * w_val[1] + w_val[0])
plt.show()

