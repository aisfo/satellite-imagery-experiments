import tensorflow as tf
import numpy as np



pos_weight = tf.placeholder_with_default(1.0, ())
input_image = tf.placeholder(tf.float32, shape=(1, 1500, 1500, 3))
label_image = tf.placeholder(tf.float32, shape=(1, 1500, 1500, 1))


global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.001, global_step, 100, 0.9, staircase=True)
tf.summary.scalar('learning_rate', learning_rate)



conv0_w = tf.get_variable("conv0_w", [16, 16, 3, 32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
conv0_b = tf.get_variable("conv0_b", [32], initializer=tf.random_normal_initializer(0))
reg = tf.nn.l2_loss(conv0_w)

test = tf.nn.conv2d(test, conv0_w, [1, 4, 4, 1], padding='SAME')
test = tf.nn.relu(test + conv0_b)

tf.summary.histogram("conv0", test)
tf.summary.histogram("conv0_w", conv0_w)


conv1_w = tf.get_variable("conv1_w", [3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
conv1_b = tf.get_variable("conv1_b", [32], initializer=tf.random_normal_initializer(0))
reg += tf.nn.l2_loss(conv1_w)

test = tf.nn.conv2d(test, conv1_w, [1, 1, 1, 1], padding='SAME')
test = tf.nn.relu(test + conv1_b)

tf.summary.histogram("conv1", test)
tf.summary.histogram("conv1_w", conv1_w)


conv2_w = tf.get_variable("conv2_w", [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
conv2_b = tf.get_variable("conv2_b", [64], initializer=tf.random_normal_initializer(0))
reg += tf.nn.l2_loss(conv2_w)

test = tf.nn.conv2d(test, conv2_w, [1, 1, 1, 1], padding='SAME')
test = tf.nn.relu(test + conv2_b)

tf.summary.histogram("conv2", test)
tf.summary.histogram("conv2_w", conv2_w)


conv3_w = tf.get_variable("conv3_w", [5, 5, 1, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
conv3_b = tf.get_variable("conv3_b", [1], initializer=tf.random_normal_initializer(0))

test = tf.nn.conv2d_transpose(test, conv3_w, [1, 1500, 1500, 1], [1, 4, 4, 1], padding='SAME')
test = tf.nn.relu(test + conv3_b)

tf.summary.histogram("conv3", test)
tf.summary.histogram("conv3_w", conv3_w)



error = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(test, labels, pos_weight)) / (1500 * 1500 - 1) 
tf.summary.scalar('error', error)

full_error = error + 0.001 * reg
tf.summary.scalar('full_error', full_error)

train = tf.train.AdamOptimizer(learning_rate).minimize(full_error, global_step=global_step)



test = tf.nn.sigmoid(test)

tf.summary.image('input_image', input_image)
tf.summary.image('label_image', label_image)
tf.summary.image('predicted_image', test)


summary = tf.summary.merge_all()