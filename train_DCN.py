#!/usr/bin/env python

import os
from datetime import datetime
import time
from six.moves import xrange

import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.metrics

import tensorflow.examples.tutorials.mnist.input_data as input_data
from data import cifar10, cifar100, mnist
from networks import DCN_fc, DCN_conv

# Dataset Configuration
tf.app.flags.DEFINE_string('dataset', 'mnist', """Dataset type.""")
tf.app.flags.DEFINE_string('data_dir', './data/mnist/', """Path to the dataset.""")
tf.app.flags.DEFINE_integer('num_classes', 10, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_train_instance', 60000, """Number of training images.""")
tf.app.flags.DEFINE_integer('num_test_instance', 10000, """Number of test images.""")

# Network Configuration
tf.app.flags.DEFINE_string('network', 'DCN-fc', """Network architecture""")
tf.app.flags.DEFINE_boolean('fc_bias', True, """Whether to add bias after fc multiply""")
tf.app.flags.DEFINE_integer('batch_size', 100, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_clusters', 3, """Number of clusters """)
tf.app.flags.DEFINE_integer('feature_dim', 10, """The dimension of features""")

# Optimization Configuration
tf.app.flags.DEFINE_float('clustering_loss', 1.0, """The weight of clustering loss""")
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
tf.app.flags.DEFINE_string('lr_step_epoch', "100.0", """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")

# Training Configuration
tf.app.flags.DEFINE_string('train_dir', './train', """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 120000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_integer('test_interval', 600, """Number of iterations to run a test""")
tf.app.flags.DEFINE_integer('test_iter', 100, """Number of iterations during a test""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 10000, """Number of iterations to save parameters as a checkpoint""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.96, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_string('checkpoint', None, """Model checkpoint to load""")

FLAGS = tf.app.flags.FLAGS

pd.set_option('precision', 4)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


def NMI(cluster_val, label_val):
    return sklearn.metrics.normalized_mutual_info_score(cluster_val, label_val)


def train():
    print('[Dataset Configuration]')
    print('\tDataset: %s' % FLAGS.dataset)
    print('\tDataset dir: %s' % FLAGS.data_dir)
    print('\tNumber of classes: %d' % FLAGS.num_classes)
    print('\tNumber of training images: %d' % FLAGS.num_train_instance)
    print('\tNumber of test images: %d' % FLAGS.num_test_instance)

    print('[Network Configuration]')
    print('\tNetwork architecture: %s' % FLAGS.network)
    print('\tFC layer bias: %d' % FLAGS.fc_bias)
    print('\tBatch size: %d' % FLAGS.batch_size)
    print('\tThe number of clusters: %d' % FLAGS.num_clusters)
    print('\tThe feature dimension: %d' % FLAGS.feature_dim)

    print('[Optimization Configuration]')
    print('\tClustering loss weight: %f' % FLAGS.clustering_loss)
    print('\tL2 loss weight: %f' % FLAGS.l2_weight)
    print('\tThe momentum optimizer: %f' % FLAGS.momentum)
    print('\tInitial learning rate: %f' % FLAGS.initial_lr)
    print('\tEpochs per lr step: %s' % FLAGS.lr_step_epoch)
    print('\tLearning rate decay: %f' % FLAGS.lr_decay)

    print('[Training Configuration]')
    print('\tTrain dir: %s' % FLAGS.train_dir)
    print('\tTraining max steps: %d' % FLAGS.max_steps)
    print('\tSteps per displaying info: %d' % FLAGS.display)
    print('\tSteps per testing: %d' % FLAGS.test_interval)
    print('\tSteps during testing: %d' % FLAGS.test_iter)
    print('\tSteps per saving checkpoints: %d' % FLAGS.checkpoint_interval)
    print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    print('\tLog device placement: %d' % FLAGS.log_device_placement)


    with tf.Graph().as_default():
        init_step = 0
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and labels
        if 'cifar-10'==FLAGS.dataset:
            with tf.device('/CPU:0'):
                with tf.variable_scope('train_image'):
                    train_images, train_labels = cifar10.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=True)
                with tf.variable_scope('test_image'):
                    test_images, test_labels = cifar10.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=False)
        elif 'cifar-100'==FLAGS.dataset:
            with tf.device('/CPU:0'):
                with tf.variable_scope('train_image'):
                    train_images, train_labels = cifar100.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=True)
                with tf.variable_scope('test_image'):
                    test_images, test_labels = cifar100.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=False)
        elif 'mnist'==FLAGS.dataset:
            # Tensorflow default dataset
            mnist_read = input_data.read_data_sets(FLAGS.data_dir, one_hot=False, validation_size=0)
            def train_func():
                return mnist_read.train.next_batch(FLAGS.batch_size, shuffle=True)
            def test_func():
                return mnist_read.test.next_batch(FLAGS.batch_size, shuffle=False)
            train_images, train_labels = tf.py_func(train_func, [], [tf.float32, tf.uint8])
            train_images.set_shape([FLAGS.batch_size, 784])
            train_labels.set_shape([FLAGS.batch_size])
            train_labels = tf.cast(train_labels, tf.int32)
            test_images, test_labels = tf.py_func(test_func, [], [tf.float32, tf.uint8])
            test_images.set_shape([FLAGS.batch_size, 784])
            test_labels.set_shape([FLAGS.batch_size])
            test_labels = tf.cast(test_labels, tf.int32)
        elif 'mnist-aug'==FLAGS.dataset:
            with tf.device('/CPU:0'):
                with tf.variable_scope('train_image'):
                    train_images, train_labels = mnist.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=True)
                with tf.variable_scope('test_image'):
                    test_images, test_labels = mnist.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=False)


        # Build model
        if 'DCN-fc'==FLAGS.network:
            network = DCN_fc
        if 'DCN-conv'==FLAGS.network:
            network = DCN_conv

        # 1) Training Network
        hp = network.HParams(batch_size=FLAGS.batch_size,
                             num_classes=FLAGS.num_classes,
                             num_clusters=FLAGS.num_clusters,
                             feature_dim=FLAGS.feature_dim,
                             fc_bias=FLAGS.fc_bias,
                             clustering_loss=FLAGS.clustering_loss,
                             weight_decay=FLAGS.l2_weight,
                             momentum=FLAGS.momentum)
        network_train = network.LeNet(hp, train_images, train_labels, global_step, name='train')
        network_train.build_model()
        network_train.build_network_train_op()
        network_train.build_centroid_update_op()

        train_summary_op = tf.summary.merge_all()  # Summaries(training)

        # 2) Test network(reuse_variables!)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            scope.reuse_variables()
            network_test = network.LeNet(hp, test_images, test_labels, global_step, name='test')
            network_test.build_model()

        # Learning rate decay
        lr_decay_steps = [float(s) for s in FLAGS.lr_step_epoch.split(',')]
        lr_decay_steps = [int(f) for f in [s*FLAGS.num_train_instance/FLAGS.batch_size for s in lr_decay_steps]]
        def get_lr(initial_lr, lr_decay, lr_decay_steps, global_step):
            lr = initial_lr
            for s in lr_decay_steps:
                if global_step >= s:
                    lr *= lr_decay
            return lr

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
        if FLAGS.checkpoint is not None:
            saver.restore(sess, FLAGS.checkpoint)
            init_step = global_step.eval(session=sess)
            print('Load checkpoint %s' % FLAGS.checkpoint)
        else:
            print('No checkpoint file of basemodel found. Start from the scratch.')

        # Start queue runners & summary_writer
        tf.train.start_queue_runners(sess=sess)
        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)
        summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, str(global_step.eval(session=sess))))

        # Training!
        test_best_acc = 0.0
        train_cont_table, test_cont_table = np.zeros((FLAGS.num_clusters, FLAGS.num_classes), np.int32), np.zeros((FLAGS.num_clusters, FLAGS.num_classes), np.int32)
        train_clusters_val, train_labels_val, test_clusters_val, test_labels_val = [], [], [], []
        for step in xrange(init_step, FLAGS.max_steps):
            # Test
            if step % FLAGS.test_interval == 0:
                test_loss, test_recon_loss, test_cluster_loss = 0.0, 0.0, 0.0
                for i in range(FLAGS.test_iter):
                    network_loss_val, recon_loss_val, cluster_loss_val, assigns_val, labels_val, cent_cnts_val = sess.run([network_test.network_loss, network_test.recon_loss, network_test.cluster_loss, network_test._assigns, test_labels, network_test._centroid_cnts],
                            feed_dict={network_test.is_train:False})
                    test_loss += network_loss_val
                    test_recon_loss += recon_loss_val
                    test_cluster_loss += cluster_loss_val
                    test_clusters_val.extend(assigns_val)
                    test_labels_val.extend(labels_val)
                    for j in range(FLAGS.batch_size):
                        test_cont_table[assigns_val[j]][labels_val[j]] += 1
                test_loss /= FLAGS.test_iter
                test_recon_loss /= FLAGS.test_iter
                test_cluster_loss /= FLAGS.test_iter
                format_str = ('%s: (Test)     step %d, loss=%.4f(%.4f, %.4f)')
                print (format_str % (datetime.now(), step, test_loss, test_recon_loss, test_cluster_loss))

                test_summary = tf.Summary()
                test_summary.value.add(tag='test/loss', simple_value=test_loss)
                test_summary.value.add(tag='test/recon_loss', simple_value=test_recon_loss)
                test_summary.value.add(tag='test/cluster_loss', simple_value=test_cluster_loss)
                summary_writer.add_summary(test_summary, step)
                summary_writer.flush()

            # Train
            lr_value = get_lr(FLAGS.initial_lr, FLAGS.lr_decay, lr_decay_steps, step)
            start_time = time.time()

            if step < 30000:  # Pretrain. reconstruction only
                _, lr_value, loss_val, recon_loss_val, cluster_loss_val, assigns_val, labels_val, train_summary_str = \
                        sess.run([network_train.pretrain_op, network_train.lr, network_train.network_loss, network_train.recon_loss, network_train.cluster_loss, network_train._assigns, train_labels, train_summary_op],
                            feed_dict={network_train.is_train:True, network_train.lr:lr_value})
            else:
                _, _, lr_value, loss_val, recon_loss_val, cluster_loss_val, assigns_val, labels_val, train_summary_str = \
                        sess.run([network_train.train_op, network_train.centroid_update_op, network_train.lr, network_train.network_loss, network_train.recon_loss, network_train.cluster_loss, network_train._assigns, train_labels, train_summary_op],
                            feed_dict={network_train.is_train:True, network_train.lr:lr_value})
            duration = time.time() - start_time

            assert not np.isnan(loss_val)
            for j in range(FLAGS.batch_size):
                train_cont_table[assigns_val[j]][labels_val[j]] += 1
            train_clusters_val.extend(assigns_val)
            train_labels_val.extend(labels_val)

            # Display & Summary(training)
            if step % FLAGS.display == 0 or step < 10:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: (Training) step %d, loss=%.4f(%.4f, %.4f), lr=%f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_val, recon_loss_val, cluster_loss_val, lr_value,
                                     examples_per_sec, sec_per_batch))
                summary_writer.add_summary(train_summary_str, step)


            # Display clustering status
            if step % FLAGS.test_interval == 0:
                print('')
                print('NMI: train %.4f, test %.4f' % (NMI(train_clusters_val, train_labels_val), NMI(test_clusters_val, test_labels_val)))
                print('Train clustering / Test clustering')
                a = pd.DataFrame(train_cont_table)
                b = pd.DataFrame(test_cont_table)
                c = pd.DataFrame(['|'] * FLAGS.num_clusters, columns=[''])
                d = pd.DataFrame(np.expand_dims(cent_cnts_val, axis=1))
                print(pd.concat([a, c, b, c, d], axis=1))

                train_cont_table[:] = 0
                test_cont_table[:] = 0
                train_clusters_val, train_labels_val, test_clusters_val, test_labels_val = [], [], [], []

            # Save the model checkpoint periodically.
            if (step > init_step and step % FLAGS.checkpoint_interval == 0) or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)

        # After training, collect embedding vectors and images of the test data
        from skimage.io import imsave
        from tensorflow.contrib.tensorboard.plugins import projector
        num_embed_batch = 4 * 4
        print('Making embedding of %d test images' % (num_embed_batch * FLAGS.batch_size))
        embed_images, embed_labels, embed_features = np.zeros((4*10*28, 4*10*28, 4), dtype=np.uint8), [], []
        cnt = 0
        for i in range(num_embed_batch):
            features_val, images_val, labels_val = sess.run([network_test._features, test_images, test_labels],
                                                            feed_dict={network_test.is_train:False})
            for j in range(FLAGS.batch_size):
                r, c = cnt % (4*10), cnt // (4*10)
                embed_images[c*28:(c+1)*28,r*28:(r+1)*28] = \
                    np.concatenate([255-np.tile(255*images_val[j], [1, 1, 3]), np.ones([28, 28, 1])*255], axis=2)
                cnt += 1
            embed_features.append(features_val)
            embed_labels.append(labels_val)

        # Close session to create new session for projector
        sess.close()

    # Save embedding
    with tf.Graph().as_default():
        projector_dir = os.path.join(FLAGS.train_dir, 'projector')
        os.makedirs(projector_dir)
        sess2 = tf.InteractiveSession()

        imsave(projector_dir + '/sprite.png', embed_images)

        embed_features = np.concatenate(embed_features, axis=0)
        embed_var = tf.Variable(embed_features, trainable=False, name='embedding')
        sess2.run(tf.initialize_variables([embed_var]))

        embed_labels = np.concatenate(embed_labels, axis=0)
        with open(projector_dir + '/metadata.tsv', 'w') as f:
            for l in embed_labels:
                f.write('{}\n'.format(l))

        writer = tf.summary.FileWriter(projector_dir, sess2.graph)

        config = projector.ProjectorConfig()
        embed= config.embeddings.add()
        embed.tensor_name = 'embedding:0'
        embed.metadata_path = os.path.join('metadata.tsv')
        embed.sprite.image_path = os.path.join('sprite.png')
        embed.sprite.single_image_dim.extend([28, 28])
        projector.visualize_embeddings(writer, config)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
        saver.save(sess2, projector_dir + '/a_model.ckpt', global_step=step)

def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
