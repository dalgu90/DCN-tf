import os

import numpy as np
import tensorflow as tf

from skimage.io import imsave
from tensorflow.contrib.tensorboard.plugins import projector

# TensorFlow helper functions

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'
GATING_KEY = 'GATING'

def _fc(x, out_dim, bias=True, name='fc'):
    with tf.variable_scope(name):
        # Main operation: fc
        w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                        tf.float32, initializer=tf.random_normal_initializer(
                            stddev=np.sqrt(1.0/x.get_shape().as_list()[1])))
        if w not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, w)
        if not bias:
            fc = tf.matmul(x, w)
        else:
            b = tf.get_variable('biases', [out_dim], tf.float32,
                                initializer=tf.constant_initializer(0.0))
            fc = tf.nn.bias_add(tf.matmul(x, w), b)
    return fc

def _relu(x, leakness=0.0, name=None):
    if leakness > 0.0:
        name = 'lrelu' if name is None else name
        return tf.maximum(x, x*leakness, name='lrelu')
    else:
        name = 'relu' if name is None else name
        return tf.nn.relu(x, name='relu')

def _relu_group(inputs, leakness=0.0, name='relu_group'):
    outputs = []
    for i, x in enumerate(inputs):
        if x is not None:
            relu = _relu(x, leakness, 'relu_%d'%(i+1))
            outputs.append(relu)
        else:
            outputs.append(None)
    return outputs

def _conv(x, filter_size, out_channel, strides, pad='SAME', name='conv'):
    in_shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        # Main operation: conv2d
        kernel = tf.get_variable('kernel', [filter_size, filter_size, in_shape[3], out_channel],
                        tf.float32, initializer=tf.random_normal_initializer(
                            stddev=np.sqrt(1.0/filter_size/filter_size/in_shape[3])))
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
        conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], pad)
    return conv

def _deconv(x, filter_size, out_channel, strides, pad='SAME', name='deconv'):
    in_shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        # Main operation: conv2d_transpose
        kernel = tf.get_variable('kernel', [filter_size, filter_size, out_channel, in_shape[3]],
                                 tf.float32, initializer=tf.random_normal_initializer(
                                 stddev=np.sqrt(1.0/filter_size/filter_size/in_shape[3])))
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
        if 'VALID' == pad:
            h = in_shape[1] * strides + filter_size - strides
            w = in_shape[2] * strides + filter_size - strides
        elif 'SAME' == pad:
            h = in_shape[1] * strides
            w = in_shape[2] * strides
        deconv = tf.nn.conv2d_transpose(x, kernel, [in_shape[0], w, h, out_channel],
                                        [1, strides, strides, 1], pad)
    return deconv

def _bn(x, is_train, global_step=None, name='bn', no_scale=False):
    moving_average_decay = 0.9
    with tf.variable_scope(name):
        decay = moving_average_decay
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                        initializer=tf.zeros_initializer(), trainable=False)
        sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                        initializer=tf.ones_initializer(), trainable=False)
        beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                        initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                        initializer=tf.ones_initializer(), trainable=(not no_scale))
        # BN when training
        update = 1.0 - decay
        # with tf.control_dependencies([tf.Print(decay, [decay])]):
            # update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_sigma = sigma.assign_sub(update*(sigma - batch_var))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

        mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
                            lambda: (mu, sigma))
        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    return bn

def _dropout(x, keep_prob=1.0, name=None):
    assert keep_prob >= 0.0 and keep_prob <= 1.0
    if keep_prob == 1.0:
        return x
    else:
        return tf.nn.dropout(x, keep_prob, name=name)

def _fc_with_init(x, out_dim, bias=True, init_w=None, init_b=None, trainable=True, name='fc'):
    with tf.variable_scope(name):
        # Main operation: fc
        if init_w is not None:
            initializer_w = tf.constant_initializer(init_w)
        else:
            initializer_w = tf.random_normal_initializer(stddev=np.sqrt(1.0/x.get_shape().as_list()[1]))

        w = tf.get_variable('weights', [x.get_shape()[1], out_dim], tf.float32,
                            initializer=initializer_w, trainable=trainable)
        if trainable and (w not in tf.get_collection(WEIGHT_DECAY_KEY)):
            tf.add_to_collection(WEIGHT_DECAY_KEY, w)

        fc = tf.matmul(x, w)

        if bias:
            if init_b is not None:
                initializer_b = tf.constant_initializer(init_b)
            else:
                initializer_b = tf.constant_initializer(0.0)

            b = tf.get_variable('biases', [out_dim], tf.float32,
                                initializer=initializer_b, trainable=trainable)
            fc = tf.nn.bias_add(fc, b)

    return fc

def save_embedding_projector(projector_dir, embeddings, labels, images_sprites, image_dim):
    metadata_fname = 'metadata.tsv'
    image_fname = 'sprite.png'
    ckpt_fname = 'model.ckpt'

    if not os.path.exists(projector_dir):
        os.makedirs(projector_dir)

    # Save labels
    with open(os.path.join(projector_dir, metadata_fname), 'w') as fd:
        fd.write(''.join(['%d\n' % l for l in labels]))

    # Save images
    imsave(os.path.join(projector_dir, image_fname), images_sprites)

    # Save embeddings inside new graph scope
    with tf.Graph().as_default() as g:
        sess = tf.InteractiveSession(graph=g)

        embed_var = tf.Variable(embeddings, trainable=False, name='embedding')
        sess.run(tf.initialize_variables([embed_var]))

        writer = tf.summary.FileWriter(projector_dir, sess.graph)

        # Projector
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'embedding:0'
        embed.metadata_path = metadata_fname
        embed.sprite.image_path = image_fname
        embed.sprite.single_image_dim.extend(image_dim)
        projector.visualize_embeddings(writer, config)

        saver = tf.train.Saver([embed_var])
        saver.save(sess, os.path.join(projector_dir, ckpt_fname), global_step=0)

        sess.close()
