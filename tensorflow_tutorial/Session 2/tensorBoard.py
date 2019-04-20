import tensorflow as tf

# create graph
a = tf.constant(2)
b = tf.constant(3)
# e=tf.constant(6)
c = tf.add(a, b)
# d=tf.multiply(c,e)
# launch the graph in a session
# with tf.Session() as sess:
#     print(sess.run(c))
# To visualize the program with TensorBoard, we need to write log files of the program. To write event files,
#  we first need to create a writer for those logs, using this code:

# creating the writer out of the session
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
# tensorboard --logdir ./simple_tensorboard
# launch the graph in a session
with tf.Session() as sess:
    # or creating the writer inside the session
    writer = tf.summary.FileWriter('./simple_tensorboard', sess.graph)
    print(sess.run(c))

