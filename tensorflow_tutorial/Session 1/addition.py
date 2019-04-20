import os
import tensorflow as tf
#Turn Off Warning
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

#Define computational Graph
# different between placeholder and variable in tensorflow

# Placeholder simply allocates block of memory for future use.
#  Later, we can use feed_dict to feed the data into placeholder.
# By default, placeholder has an unconstrained shape, which allows you to feed tensors of different shapes in a session.
#  You can make constrained shape by passing optional argument -shape
X=tf.placeholder(tf.float32,name="XV")
Y=tf.placeholder(tf.float32,name="YV")

addition=tf.add(X,Y,name="adder")
#Create the session
with tf.Session() as session:
    resualt=session.run(addition,feed_dict={X:[1],Y:[4]})
    resualt = session.run(addition, feed_dict={X: [1 ,2 ,3 ], Y: [4,3 ,3]})
    print(resualt)