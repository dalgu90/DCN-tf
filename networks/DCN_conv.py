#!/usr/bin/env python

from collections import namedtuple

import tensorflow as tf
import numpy as np

from .network import Network
import utils

HParams = namedtuple('HParams',
                    'batch_size, num_classes, num_clusters, feature_dim, fc_bias, clustering_loss, weight_decay, momentum')
HParams.__new__.__defaults__ = (100, 10, 3, 10, True, 1.0, 0.0001, 0.9)

class LeNet(Network):
    def __init__(self, hp, images, labels, global_step, name='lenet'):
        super(LeNet, self).__init__(hp, images, labels, global_step, name)

    def build_model(self):
        print('Building model')

        with tf.name_scope(self._name+('' if self._name.endswith('/') else '/')+'encoder'):
            # Input reshaping
            # x = tf.reshape(self._images, [self._hp.batch_size, -1])
            # input_dim = x.get_shape().as_list()[-1]
            x = self._images

            # Encoder
            conv_filters_enc = [20, 20]
            fc_filters_enc = [200]
            with tf.variable_scope('encoder'):
                for i, f in enumerate(conv_filters_enc):
                    x = self._conv(x, 3, f, 1, pad='SAME', name='conv_%d'%(i+1))
                    x = self._max_pool(x, 2, 2, name="pool_%d"%(i+1))
                last_conv_dim = x.get_shape().as_list()
                x = tf.reshape(x, [self._hp.batch_size, -1])
                first_fc_dim = x.get_shape().as_list()
                for i, f in enumerate(fc_filters_enc):
                    x = self._fc(x, f, bias=self._hp.fc_bias, name='fc_%d'%(i+1))
                    x = self._relu(x, name='relu_%d'%(i+1))
                x = self._fc(x, self._hp.feature_dim, name='fc_%d'%(len(fc_filters_enc)+1))
            self._features = x

            # Clustering
            with tf.variable_scope('cluster'):
                centroid_init_val = np.random.normal(0.0, 1.0, (self._hp.num_clusters, self._hp.feature_dim))
                self._centroids = tf.get_variable('centroids', shape=(self._hp.num_clusters, self._hp.feature_dim), dtype=tf.float32,
                                                 initializer=tf.constant_initializer(centroid_init_val))
                self._centroid_cnts = tf.get_variable('centroid_cnts', shape=(self._hp.num_clusters), dtype=tf.int32,
                                                      initializer=tf.ones_initializer(), trainable=False)
                self._cluster_distsqs = tf.reduce_sum(tf.square(tf.tile(tf.expand_dims(self._features, axis=1), multiples=[1, self._hp.num_clusters, 1]) - \
                    tf.tile(tf.expand_dims(self._centroids, axis=0), multiples=[self._hp.batch_size, 1, 1])), axis=2)
                self._assigns = tf.argmin(self._cluster_distsqs, axis=1)

            # Decoder
            conv_filters_dec = conv_filters_enc[-2::-1]  # reverse the number of filters
            fc_filters_dec = fc_filters_enc[::-1]  # reverse the number of filters
            with tf.variable_scope('decoder'):
                for i, f in enumerate(fc_filters_dec):
                    x = self._fc(x, f, bias=self._hp.fc_bias, name='fc_%d'%(i+1))
                    x = self._relu(x, name='relu_%d'%(i+1))
                x = self._fc(x, first_fc_dim[1], name='fc_%d'%(len(fc_filters_dec)+1))
                x = tf.reshape(x, last_conv_dim)
                for i, f in enumerate(conv_filters_dec):
                    x = self._deconv(x, 3, f, 2, pad='SAME', name='deconv_%d'%(i+1))
                x = self._deconv(x, 3, 1, 2, pad='SAME', name='deconv_%d'%(len(conv_filters_dec)+1))
                x = self._relu(x, name='relu_%d'%(len(fc_filters_dec)+1))

            # Reconstruction reshaping
            self._recons = x
            # self._recons = tf.reshape(x, self._images.get_shape())

            # Loss & acc
            self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self._recons - self._images), axis=[1, 2, 3]))
            self.cluster_loss = self._hp.clustering_loss * tf.reduce_mean(tf.reduce_min(self._cluster_distsqs, axis=1))
            self.network_loss = self.recon_loss + self.cluster_loss
            tf.summary.scalar('network_loss', self.network_loss)
            tf.summary.scalar('recon_loss', self.recon_loss)
            tf.summary.scalar('cluster_loss', self.cluster_loss)

            tf.summary.image('images', self._images[:3])
            tf.summary.image('recons', self._recons[:3])


    def build_network_train_op(self):
        print('Build training ops')

        with tf.name_scope(self._name+('' if self._name.endswith('/') else '/')):
            # Learning rate
            tf.summary.scalar('learing_rate', self.lr)

            losses = [self.network_loss]
            losses_pretrain = [self.recon_loss]

            # Add l2 loss
            with tf.variable_scope('l2_loss'):
                costs = [tf.nn.l2_loss(var) for var in tf.get_collection(utils.WEIGHT_DECAY_KEY)]
                l2_loss = tf.multiply(self._hp.weight_decay, tf.add_n(costs))
                losses.append(l2_loss)
                losses_pretrain.append(l2_loss)

            self._total_loss = tf.add_n(losses)
            self._pretrain_loss = tf.add_n(losses_pretrain)

            # Gradient descent step
            opt = tf.train.MomentumOptimizer(self.lr, self._hp.momentum)
            grads_and_vars = opt.compute_gradients(self._total_loss, tf.trainable_variables())
            apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self._global_step)
            self.train_op = apply_grad_op

            grads_and_vars = opt.compute_gradients(self._pretrain_loss, tf.trainable_variables())
            apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self._global_step)
            self.pretrain_op = apply_grad_op

            # Batch normalization moving average update
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # if update_ops:
                # with tf.control_dependencies(update_ops+[apply_grad_op]):
                    # self.train_op = tf.no_op()
            # else:
                # self.train_op = apply_grad_op


    def build_centroid_update_op(self):
        print('Build cluster centroid update ops')

        with tf.device('/CPU:0'):
            with tf.name_scope(self._name+('' if self._name.endswith('/') else '/')):
                update_op = tf.no_op()

                with tf.name_scope('centroid_update'):
                    for i in range(self._hp.batch_size):
                        assign = self._assigns[i]
                        cnt = self._centroid_cnts[assign]
                        centroid = self._centroids[assign]
                        feature = self._features[i]

                        # Update centroid
                        with tf.control_dependencies([update_op]):
                            new_centroid = centroid - (1.0/tf.cast(cnt, tf.float32))*(centroid - feature)
                            centroid_update_op = tf.scatter_update(self._centroids, assign, new_centroid)

                        # Update cnt
                        with tf.control_dependencies([centroid_update_op]):
                            update_op = tf.scatter_update(self._centroid_cnts, assign, cnt + 1)

                    self.centroid_update_op = update_op


