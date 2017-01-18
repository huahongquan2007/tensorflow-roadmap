import tensorflow as tf

filenames = tf.train.match_filenames_once('/home/ubuntu/working_data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/*.WAV')
count_num_files = tf.size(filenames)
filename_queue = tf.train.string_input_producer(filenames)

reader = tf.WholeFileReader()
filename, file_contents = reader.read(filename_queue)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_files = sess.run(count_num_files)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(num_files):
        audio_file = sess.run(filename)
        print(audio_file)