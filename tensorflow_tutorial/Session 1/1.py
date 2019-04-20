import tensorflow as tf







# print(a)
import os
import tensorflow as tf
#Turn Off Warning
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

a=tf.add(3,5)
# sess=tf.Session()
# print(sess.run(a))
# sess.close()
with tf.Session() as sess:
  print(sess.run(a))
#________________
# x=2
# y=3
# op1=tf.add(x,y)
# op2=tf.multiply(x,y)
# op3=tf.pow(op1,op2)
# with tf.Session() as sess:
#   print(sess.run(op3))
