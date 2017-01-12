import tensorflow as tf
import numpy as np

"""
    Usage:
    mkdir logs
    python tensorboard_example.py
    tensorboard --logdir=./logs
"""

raw_data = np.random.normal(10, 1, 100)

alpha = tf.constant(0.05)
curr_value = tf.placeholder(tf.float32)
prev_avg = tf.Variable(0.)

update_avg = alpha * curr_value + (1 - alpha) * prev_avg

avg_hist = tf.summary.scalar("running average", update_avg)
value_hist = tf.summary.scalar("incoming values", curr_value)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs")

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    writer.add_graph(sess.graph)

    for i in range(len(raw_data)):
        summary_str, curr_avg = sess.run([merged, update_avg], feed_dict={ curr_value : raw_data[i] })
        sess.run(tf.assign(prev_avg, curr_avg))
        print(raw_data[i], curr_avg)
        writer.add_summary(summary_str, i)