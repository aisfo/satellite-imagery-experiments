import tensorflow as tf
import numpy as np




input_image = tf.placeholder(tf.float32, shape=(1, 1500, 1500, 3))
label_image = tf.placeholder(tf.float32, shape=(1, 1500, 1500, 1))
is_train = tf.placeholder_with_default(True, ())


global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.0005, global_step, 50, 0.99, staircase=True)
tf.summary.scalar('learning_rate', learning_rate)



conv0_w = tf.get_variable("conv0_w", [16, 16, 3, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
conv0_b = tf.get_variable("conv0_b", [64], initializer=tf.random_normal_initializer(0))
reg = tf.nn.l2_loss(conv0_w)

test = tf.nn.conv2d(input_image, conv0_w, [1, 4, 4, 1], padding='SAME')
#test = batch_norm(test, 64, is_train)
test = tf.nn.relu(test + conv0_b)

tf.summary.histogram("conv0", test)
tf.summary.histogram("conv0_w", conv0_w)

test = tf.nn.max_pool(test, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

conv1_w = tf.get_variable("conv1_w", [4, 4, 64, 128], initializer=tf.contrib.layers.xavier_initializer_conv2d())
conv1_b = tf.get_variable("conv1_b", [128], initializer=tf.random_normal_initializer(0))
reg += tf.nn.l2_loss(conv1_w)

test = tf.nn.conv2d(test, conv1_w, [1, 1, 1, 1], padding='SAME')
#test = batch_norm(test, 128, is_train)
test = tf.nn.relu(test + conv1_b)

tf.summary.histogram("conv1", test)
tf.summary.histogram("conv1_w", conv1_w)


conv2_w = tf.get_variable("conv2_w", [3, 3, 128, 80], initializer=tf.contrib.layers.xavier_initializer_conv2d())
conv2_b = tf.get_variable("conv2_b", [80], initializer=tf.random_normal_initializer(0))
reg += tf.nn.l2_loss(conv2_w)

test = tf.nn.conv2d(test, conv2_w, [1, 1, 1, 1], padding='SAME')
#test = batch_norm(test, 80, is_train)
test = tf.nn.relu(test + conv2_b)

tf.summary.histogram("conv2", test)
tf.summary.histogram("conv2_w", conv2_w)


conv3_w = tf.get_variable("conv3_w", [16, 16, 16, 80], initializer=tf.contrib.layers.xavier_initializer_conv2d())
conv3_b = tf.get_variable("conv3_b", [16], initializer=tf.random_normal_initializer(0))
reg += tf.nn.l2_loss(conv3_w)

test = tf.nn.conv2d_transpose(test, conv3_w, [1, 750, 750, 16], [1, 2, 2, 1], padding='SAME')
#test = batch_norm(test, 16, is_train)
test = tf.nn.relu(test + conv3_b)

tf.summary.histogram("conv3", test)
tf.summary.histogram("conv3_w", conv3_w)


conv4_w = tf.get_variable("conv4_w", [16, 16, 1, 16], initializer=tf.contrib.layers.xavier_initializer_conv2d())
conv4_b = tf.get_variable("conv4_b", [1], initializer=tf.random_normal_initializer(0))
reg += tf.nn.l2_loss(conv4_w)

test = tf.nn.conv2d_transpose(test, conv4_w, [1, 1500, 1500, 1], [1, 2, 2, 1], padding='SAME')
test = test + conv4_b

tf.summary.histogram("conv4", test)
tf.summary.histogram("conv4_w", conv4_w)



error = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(test, label_image)) / (1500 * 1500 - 1) 
tf.summary.scalar('error', error)

full_error = error + 0.0005 * reg
tf.summary.scalar('full_error', full_error)

train = tf.train.AdamOptimizer(learning_rate).minimize(full_error, global_step=global_step)



test = tf.nn.sigmoid(test)

# tf.summary.image('input_image', input_image)
# tf.summary.image('label_image', label_image)
# tf.summary.image('predicted_image', test)


summary = tf.summary.merge_all()