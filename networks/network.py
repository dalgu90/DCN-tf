#!/usr/bin/env python

import tensorflow as tf
import numpy as np

import utils


class Network(object):
    def __init__(self, hp, images, labels, global_step, name='network'):
        self._hp = hp # Hyperparameters
        self._images = images # Input image
        self._labels = labels
        self._global_step = global_step
        self._name = name
        self.lr = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)
        self._counted_scope = []
        self._flops = 0
        self._weights = 0

    def build_model(self):
        pass

    def build_train_op(self):
        pass

    # Helper functions(counts FLOPs and number of weights)
    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv(x, filter_size, out_channel, stride, pad, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        print('%s: %s' % (name, str(x.get_shape().as_list())))
        return x

    def _deconv(self, x, filter_size, out_channel, stride, pad="SAME", name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._deconv(x, filter_size, out_channel, stride, pad, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        print('%s: %s' % (name, str(x.get_shape().as_list())))
        return x

    def _fc(self, x, out_dim, bias=True, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc(x, out_dim, bias, name)
        f = 2 * (in_dim + 1) * out_dim if bias else 2 * in_dim * out_dim
        w = (in_dim + 1) * out_dim if bias else in_dim * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        print('%s: %s' % (name, str(x.get_shape().as_list())))
        return x

    def _bn(self, x, name="bn", no_scale=False):
        x = utils._bn(x, self.is_train, self._global_step, name, no_scale=no_scale)
        print('%s: %s' % (name, str(x.get_shape().as_list())))
        return x

    def _relu(self, x, name="relu"):
        x = utils._relu(x, 0.0, name)
        print('%s: %s' % (name, str(x.get_shape().as_list())))
        return x

    def _max_pool(self, x, filter_size, strides, pad="VALID", name="pool"):
        x = tf.nn.max_pool(x, [1, filter_size, filter_size, 1], [1, strides, strides, 1], padding=pad, name=name)
        print('%s: %s' % (name, str(x.get_shape().as_list())))
        return x

    def _dropout(self, x, keep_prob, name="dropout"):
        x = utils._dropout(x, keep_prob, name)
        print('%s: %s' % (name, str(x.get_shape().as_list())))
        return x

    def _get_data_size(self, x):
        return np.prod(x.get_shape().as_list()[1:])

    def _add_flops_weights(self, scope_name, f, w):
        if scope_name not in self._counted_scope:
            self._flops += f
            self._weights += w
            self._counted_scope.append(scope_name)



