
import tensorflow as tf
import numpy as np




def conv(_input, name, width, stride, out_depth, transpose=False):
    with tf.variable_scope(name):
        input_shape = _input.get_shape().as_list()
        in_depth = input_shape[-1]
        if transpose:
            conv_shape = [width, width, out_depth, in_depth]
        else:
            conv_shape = [width, width, in_depth, out_depth]

        conv_w = tf.get_variable("conv_w", conv_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_b = tf.get_variable("conv_b", out_depth, initializer=tf.random_normal_initializer(0))
         
        tf.add_to_collection("l2_losses", tf.nn.l2_loss(conv_w))

        if transpose:
            output_shape = [tf.shape(_input)[0], input_shape[1] * stride, input_shape[2] * stride, out_depth]
            _input = tf.nn.conv2d_transpose(_input, conv_w, output_shape, [1, stride, stride, 1], padding='SAME')
            _input = tf.reshape(_input, (-1, input_shape[1] * stride, input_shape[2] * stride, out_depth))
        else:
            _input = tf.nn.conv2d(_input, conv_w, [1, stride, stride, 1], padding='SAME')

        tf.summary.histogram("conv", _input)
        tf.summary.histogram("conv_w", conv_w)
        tf.summary.histogram("conv_b", conv_b)


        return (_input, conv_b)



def maxpool(_input, name, width, stride):
    with tf.variable_scope(name):
        return tf.nn.max_pool(_input, ksize=[1, width, width, 1], strides=[1, stride, stride, 1], padding='SAME')



def batch_norm(_input, name, is_train):
    with tf.variable_scope(name):
        input_shape = _input.get_shape().as_list()
        out_depth = input_shape[-1]

        offset = tf.get_variable("offset", [out_depth], initializer=tf.constant_initializer(0.0))
        scale = tf.get_variable("scale", [out_depth], initializer=tf.constant_initializer(1.0))

        tf.summary.histogram("offset", offset)
        tf.summary.histogram("scale", scale)

        batch_mean, batch_var = tf.nn.moments(_input, [0, 1, 2], name='moments')
        exp_mov_ave = tf.train.ExponentialMovingAverage(decay=0.9)
        
        def mean_var_train():
            update_exp_mov_ave = exp_mov_ave.apply([batch_mean, batch_var])
            with tf.control_dependencies([update_exp_mov_ave]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        def mean_var_test():
            return exp_mov_ave.average(batch_mean), exp_mov_ave.average(batch_var)

        mean, var = tf.cond(is_train, mean_var_train, mean_var_test)
        normed = tf.nn.batch_normalization(_input, mean, var, offset, scale, 1e-5)

        tf.summary.histogram("normed", normed)

        return normed