from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)
#_________________________________________________________________
#Intro to Convolutional Neural Networks
#Building the CNN MNIST Classifier
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional
  # Layer  # 1: Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU activation function
  #if we feed examples in batches of 100, features["x"] will contain 78,400 values, and input_layer will have a shape of [100, 28, 28, 1]
  #If filter height and width have the same value, you can instead specify a single integer for kernel_size—e.g., kernel_size=5.
  #The padding argument specifies one of two enumerated values (case-insensitive): valid (default value) or same.

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
#  out put is [batch_size, 28, 28, 32]

  # Pooling
  # Layer  # 1: Performs max pooling with a 2x2 filter and stride of 2 (which specifies that pooled regions do not overlap)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  #stride=[3, 6]

  # Convolutional
  # Layer  # 2: Applies 64 5x5 filters, with ReLU activation function

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  # Pooling
  # Layer  # 2: Again, performs max pooling with a 2x2 filter and stride of 2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
#output [batch_size, 7, 7, 64]
  # Dense Layer


  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense
  # Layer  # 1: 1,024 neurons, with dropout regularization rate of 0.4 (probability of 0.4 that any given element will be dropped during training)

  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  #To help improve the results of our model, we also apply dropout regularization to our dense layer, using the dropout method in layers:
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer

  # Dense
  # Layer  # 2 (Logits Layer): 10 neurons, one for each digit target class (0–9).
  logits = tf.layers.dense(inputs=dropout, units=10)
#Our final output tensor of the CNN, logits, has shape [batch_size, 10]
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      # the output with greater level is choosen
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
        ###################################################################################################################################
        ##  a high-level TensorFlow API that greatly simplifies machine learning programming. Estimators encapsulate the following actions:
        ##      training
        ##      evaluation
        ##      prediction
        ##      export for serving
        ####################################################################################################################################
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



# Load training and eval data
((train_data, train_labels),
 (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data/np.float32(255)
train_labels = train_labels.astype(np.int32)  # not required

eval_data = eval_data/np.float32(255)
eval_labels = eval_labels.astype(np.int32)  # not required


# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")


# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

# train one step and display the probabilties
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=1,
    hooks=[logging_hook])
mnist_classifier.train(input_fn=train_input_fn, steps=1000)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
