import tensorflow as tf
import numpy as np
from helpers import *

modelName = 'sat-res-mom-nodrop-regulup'

input_image = tf.placeholder(tf.float32, shape=(None, 1500, 1500, 3))
label_image = tf.placeholder(tf.float32, shape=(None, 1500, 1500, 1))

is_train = tf.placeholder_with_default(True, ())
global_step = tf.Variable(0, trainable=False)


norm_coef = 0.001

keep_prob = tf.cond(is_train, lambda: tf.identity(1.0), lambda: tf.identity(1.0))

learning_rate = tf.train.exponential_decay(0.1, global_step, 1500, 0.5, staircase=True)
tf.summary.scalar('learning_rate', learning_rate)



layer, bias = conv(input_image, "conv0", width=7, stride=2, out_depth=32)
layer = batch_norm(layer, 'conv0', is_train)
layer = tf.nn.relu(layer + bias)

temp = layer

layer, bias  = conv(layer, "conv1a", width=3, stride=1, out_depth=32)
layer = batch_norm(layer, 'conv1a', is_train)
layer = tf.nn.relu(layer + bias)

layer, bias  = conv(layer, "conv1b", width=3, stride=1, out_depth=32)
layer = batch_norm(layer, 'conv1b', is_train)
layer = tf.nn.relu(temp + layer + bias)

temp = layer

layer, bias  = conv(layer, "conv2a", width=3, stride=1, out_depth=32)
layer = batch_norm(layer, 'conv2a', is_train)
layer = tf.nn.relu(layer + bias)

layer, bias  = conv(layer, "conv2b", width=3, stride=1, out_depth=32)
layer = batch_norm(layer, 'conv2b', is_train)
layer = tf.nn.relu(temp + layer + bias)

temp = layer

layer, bias  = conv(layer, "conv3a", width=3, stride=1, out_depth=32)
layer = batch_norm(layer, 'conv3a', is_train)
layer = tf.nn.relu(layer + bias)

layer, bias  = conv(layer, "conv3b", width=3, stride=1, out_depth=32)
layer = batch_norm(layer, 'conv3b', is_train)
layer = tf.nn.relu(temp + layer + bias)

lay_k = layer

layer, bias  = conv(lay_k, "conv4", width=1, stride=2, out_depth=64)
layer = batch_norm(layer, 'conv4', is_train)
layer = tf.nn.relu(layer + bias)

temp = layer

layer, bias  = conv(lay_k, "conv5a", width=7, stride=2, out_depth=64)
layer = batch_norm(layer, 'conv5a', is_train)
layer = tf.nn.relu(layer + bias)

layer, bias  = conv(layer, "conv5b", width=3, stride=1, out_depth=64)
layer = batch_norm(layer, 'conv5b', is_train)
layer = tf.nn.relu(temp + layer + bias)

temp = layer

layer, bias  = conv(layer, "conv6a", width=3, stride=1, out_depth=64)
layer = batch_norm(layer, 'conv6a', is_train)
layer = tf.nn.relu(layer + bias)

layer, bias  = conv(layer, "conv6b", width=3, stride=1, out_depth=64)
layer = batch_norm(layer, 'conv6b', is_train)
layer = tf.nn.relu(temp + layer + bias)

temp = layer

layer, bias  = conv(layer, "conv7a", width=3, stride=1, out_depth=64)
layer = batch_norm(layer, 'conv7a', is_train)
layer = tf.nn.relu(layer + bias)

layer, bias  = conv(layer, "conv7b", width=3, stride=1, out_depth=64)
layer = batch_norm(layer, 'conv7b', is_train)
layer = tf.nn.relu(temp + layer + bias)

temp = layer

layer, bias  = conv(layer, "conv8a", width=3, stride=1, out_depth=64)
layer = batch_norm(layer, 'conv8a', is_train)
layer = tf.nn.relu(layer + bias)

layer, bias  = conv(layer, "conv8b", width=3, stride=1, out_depth=64)
layer = batch_norm(layer, 'conv8b', is_train)
layer = tf.nn.relu(temp + layer + bias)


layer = tf.nn.dropout(layer, keep_prob, name="dropout0")

layer, bias  = conv(layer, "conv9", width=16, stride=2, out_depth=16, transpose=True)
layer = batch_norm(layer, 'conv9', is_train)
layer = tf.nn.relu(layer + bias)

#layer = tf.concat_v2([layer, lay_k], 3)

layer, bias  = conv(layer, "conv10", width=16, stride=2, out_depth=1, transpose=True)
layer = layer + bias

result = tf.nn.sigmoid(layer)

update_ops = tf.get_collection("update_bn")

error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=layer, labels=label_image))
tf.summary.scalar('error', error)
test_summary = tf.summary.scalar('test_error',  tf.cond(is_train, lambda: tf.identity(0.0), lambda: tf.identity(error)))

full_error = error + norm_coef * tf.reduce_sum(tf.get_collection("l2_losses"))
tf.summary.scalar('full_error', full_error)

with tf.control_dependencies(update_ops):
    train = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(full_error, global_step=global_step)


# tf.summary.image('input_image', input_image)
# tf.summary.image('label_image', label_image)
# tf.summary.image('predicted_image', result)

summary = tf.summary.merge_all()
