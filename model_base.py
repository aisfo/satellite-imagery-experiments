
import tensorflow as tf
import numpy as np
from helpers import *

modelName = 'sat-base'

input_image = tf.placeholder(tf.float32, shape=(None, 1500, 1500, 3))
label_image = tf.placeholder(tf.float32, shape=(None, 1500, 1500, 1))

is_train = tf.placeholder_with_default(True, ())
global_step = tf.Variable(0, trainable=False)


norm_coef = 0.0005

keep_prob = tf.cond(is_train, lambda: tf.identity(0.5), lambda: tf.identity(1.0))

learning_rate = tf.train.exponential_decay(0.0005, global_step, 50, 0.99, staircase=True)
tf.summary.scalar('learning_rate', learning_rate)



layer, bias = conv(input_image, "conv0", width=16, stride=4, out_depth=64)
layer = tf.nn.relu(layer + bias)

layer = maxpool(layer, "pool0", width=2, stride=1)

layer, bias  = conv(layer, "conv1", width=4, stride=1, out_depth=256)
layer = tf.nn.relu(layer + bias)

layer = tf.nn.dropout(layer, keep_prob, name="dropout0")

layer, bias  = conv(layer, "conv2", width=3, stride=1, out_depth=128)
layer = tf.nn.relu(layer + bias)

layer = tf.nn.dropout(layer, keep_prob, name="dropout1")

layer, bias  = conv(layer, "conv3", width=16, stride=2, out_depth=16, transpose=True)
layer = tf.nn.relu(layer + bias)

layer, bias  = conv(layer, "conv4", width=16, stride=2, out_depth=1, transpose=True)
layer = layer + bias

result = tf.nn.sigmoid(layer)


error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(layer, label_image))
tf.summary.scalar('error', error)

full_error = error + norm_coef * sum(tf.get_collection("l2_losses"))
tf.summary.scalar('full_error', full_error)

train = tf.train.AdamOptimizer(learning_rate).minimize(full_error, global_step=global_step)


# tf.summary.image('input_image', input_image)
# tf.summary.image('label_image', label_image)
# tf.summary.image('predicted_image', result)

summary = tf.summary.merge_all()