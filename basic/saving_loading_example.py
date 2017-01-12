import tensorflow as tf

"""
Usage:
    python saving_loading_example.py
"""

sess = tf.InteractiveSession()

raw_data = [1., 2., 8., -1, 0, 5.5, 7, 13]

spikes = tf.Variable([False] * len(raw_data), name='spikes')
spikes.initializer.run()

saver = tf.train.Saver()

for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i - 1] > 5:
        spikes_val = spikes.eval()
        spikes_val[i] = True
        updater = tf.assign(spikes, spikes_val)
        updater.eval()

save_path = saver.save(sess, "/tmp/spikes.ckpt")
print("spikes data saved in file: %s" % save_path)

sess.close()

tf.reset_default_graph() # delete all variables in the current graph
sess = tf.InteractiveSession()
spikes = tf.Variable([False] * 8, name='spikes')
# spikes.initializer.run() # don't need to run this
saver = tf.train.Saver()

saver.restore(sess, "/tmp/spikes.ckpt")
print("spikes data", spikes.eval())

sess.close()