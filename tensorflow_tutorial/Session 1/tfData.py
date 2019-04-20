import tensorflow as tf
import os
#Turn Off Warning
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

dataset = tf.data.TextLineDataset("file.txt")

#You can not itrate over it
# for line in dataset:
#     print(line)

iterator = dataset.make_one_shot_iterator()
#The one_shot_iterator method creates an iterator that will be able to iterate once over the dataset.

next_element = iterator.get_next()
# you can print data
with tf.Session() as sess:
    for i in range(3):
        print(sess.run(next_element))

# #For instance, splitting words by space is as easy as adding one line
# dataset = dataset.map(lambda string: tf.string_split([string]).values)
#
# #Shuffling the dataset is also straightforward
# dataset = dataset.shuffle(buffer_size=3)
# #It will load elements 3 by 3 and shuffle them at each iteration.
#
# # create batches
# dataset = dataset.batch(2)
#
# dataset = dataset.prefetch(1)
#
# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
# with tf.Session() as sess:
#     print(sess.run(next_element))

# initial models
# # 1. For all the variables (the weights etc.)
# tf.global_variables_initializer()
#
# # 2. For the dataset, so that we can chose to move the iterator back at the beginning
# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next()
# iterator_init_op = iterator.initializer
#
# # 3. For the metrics variables, so that we can reset them to 0 at the beginning of each epoch
# metrics_init_op = tf.variables_initializer(metric_variables)



