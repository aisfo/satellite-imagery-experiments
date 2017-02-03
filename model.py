
import tensorflow as tf
import numpy as np



input_image = tf.placeholder(tf.float32, shape=(1, 1500, 1500, 3))
label_image = tf.placeholder(tf.float32, shape=(1, 1500, 1500, 1))

is_train = tf.placeholder_with_default(True, ())

global_step = tf.Variable(0, trainable=False)

learning_rate = tf.train.exponential_decay(0.0005, global_step, 50, 0.99, staircase=True)
tf.summary.scalar('learning_rate', learning_rate)

keep_prob = tf.cond(is_train, lambda: tf.identity(0.5), lambda: tf.identity(1.0))

l2_losses = 0
norm_coef = 0.0005




def conv(_input, name, width, stride, out_depth, transpose=False):
    with tf.variable_scope(name):
        input_shape = _input.get_shape().as_list()
        in_depth =input_shape[-1]
        if transpose:
            conv_shape = [width, width, out_depth, in_depth]
        else:
            conv_shape = [width, width, in_depth, out_depth]

        conv_w = tf.get_variable("conv_w",conv_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_b = tf.get_variable("conv_b", out_depth, initializer=tf.random_normal_initializer(0))
        l2_losses = tf.nn.l2_loss(conv_w)

        if transpose:
            output_shape = [input_shape[0], input_shape[1] * stride, input_shape[2] * stride, out_depth]
            _input = tf.nn.conv2d_transpose(_input, conv_w, output_shape, [1, stride, stride, 1], padding='SAME')
        else:
            _input = tf.nn.conv2d(_input, conv_w, [1, stride, stride, 1], padding='SAME')

        tf.summary.histogram("conv", _input)
        tf.summary.histogram("conv_w", conv_w)

        print(_input)

        return (_input, conv_b)


def maxpool(_input, name, width, stride):
    with tf.variable_scope(name):
        return tf.nn.max_pool(_input, ksize=[1, width, width, 1], strides=[1, stride, stride, 1], padding='SAME')



layer, bias = conv(input_image, "conv0", width=16, stride=4, out_depth=64)
layer = tf.nn.relu(layer + bias)

layer = maxpool(layer, "conv0", width=2, stride=1)

layer, bias  = conv(layer, "conv1", width=4, stride=1, out_depth=256)
layer = tf.nn.relu(layer + bias)

layer = tf.nn.dropout(layer, keep_prob, name="conv1/DropOut")

layer, bias  = conv(layer, "conv2", width=3, stride=1, out_depth=128)
layer = tf.nn.relu(layer + bias)

layer = tf.nn.dropout(layer, keep_prob, name="conv2/DropOut")

layer, bias  = conv(layer, "conv3", width=16, stride=2, out_depth=16, transpose=True)
layer = tf.nn.relu(layer + bias)

layer, bias  = conv(layer, "conv4", width=16, stride=2, out_depth=1, transpose=True)
layer = layer + bias



error = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(layer, label_image)) / (1500 * 1500 - 1) 
tf.summary.scalar('error', error)

full_error = error + norm_coef * l2_losses
tf.summary.scalar('full_error', full_error)

train = tf.train.AdamOptimizer(learning_rate).minimize(full_error, global_step=global_step)



test = tf.nn.sigmoid(layer)

# tf.summary.image('input_image', input_image)
# tf.summary.image('label_image', label_image)
# tf.summary.image('predicted_image', test)


summary = tf.summary.merge_all()