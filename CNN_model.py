import tensorflow as tf
import numpy as np
from data_helper import build_glove_dic

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_s1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_s1")
        self.input_s2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_s2")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.l2_reg_lambda = l2_reg_lambda
        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)

        self.init_weight()
        self.inference()
        self.add_dropout()
        self.add_output()
        self.add_loss_acc()

    def init_weight(self):
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            _, self.word_embedding = build_glove_dic()
            self.embedding_size = self.word_embedding.shape[1]
            self.W = tf.get_variable(name='word_embedding', shape=self.word_embedding.shape, dtype=tf.float32,
                                     initializer=tf.constant_initializer(self.word_embedding), trainable=True)
            self.s1 = tf.nn.embedding_lookup(self.W, self.input_s1)
            self.s2 = tf.nn.embedding_lookup(self.W, self.input_s2)
            self.x = tf.concat([self.s1, self.s2], axis=1)
            self.x = tf.expand_dims(self.x, -1)


    def inference(self):
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.x,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length * 2 - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

    def add_dropout(self):
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

    def add_output(self):
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[self.num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

    def add_loss_acc(self):
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.square(self.scores - self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope("pearson"):
            mid1 = tf.reduce_mean(self.scores * self.input_y) - \
                   tf.reduce_mean(self.scores) * tf.reduce_mean(self.input_y)

            mid2 = tf.sqrt(tf.reduce_mean(tf.square(self.scores)) - tf.square(tf.reduce_mean(self.scores))) * \
                   tf.sqrt(tf.reduce_mean(tf.square(self.input_y)) - tf.square(tf.reduce_mean(self.input_y)))
            self.pearson = mid1 / mid2
