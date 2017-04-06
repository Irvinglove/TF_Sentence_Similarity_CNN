#coding=utf-8
import tensorflow as tf
import numpy as np
import data_helper


# 得到权重,偏置,卷积,池化函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name='weights')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name='biases')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1],padding='SAME',name='conv')

def max_pool_22(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME',name='pool')

# 函数read_data_sets：输入文件名,返回训练集,测试集的两个句子,标签,和嵌入矩阵
# 其中,embedding_w大小为vocabulary_size × embedding_size
s1_train ,s2_train ,label_train, s1_test, s2_test, label_test, embedding_w = \
    data_helper.read_data_sets('MNIST_data')
print "得到9840维的s1_image,s2_image, label"

with  tf.name_scope('inputs'):
    x1 = tf.placeholder(tf.int64, shape=[None, 30], name='x1_input')
    x2 = tf.placeholder(tf.int64, shape=[None, 30], name='x2_input')
    y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y_input')

s1_image = tf.nn.embedding_lookup(embedding_w, x1)
s2_image = tf.nn.embedding_lookup(embedding_w, x2)
x_image = tf.matmul(s1_image,s2_image, transpose_b=True)
x_flat = tf.reshape(x_image, [-1, 30, 30, 1])

with tf.name_scope('conv_layer'):
    with tf.name_scope('weights'):
        W_conv1 = weight_variable([5, 5, 1, 2])
        tf.summary.histogram('w', W_conv1)
    with tf.name_scope('biases'):
        b_conv1 = bias_variable([2])
        tf.summary.histogram('b', b_conv1)
    with tf.name_scope('conv1'):
        h_conv1 = tf.nn.relu(conv2d(x_flat, W_conv1) + b_conv1)
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_22(h_conv1)
    with tf.name_scope('pool1_flat'):
        h_pool1_flat = tf.reshape(h_pool1, [-1, 15*15*2])

with tf.name_scope('fc1_layer'):
    with tf.name_scope('weights'):
        W_fc1 = weight_variable([15*15*2, 1024])
    with tf.name_scope('biases'):
        b_fc1 = bias_variable([1024])
    with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

with tf.name_scope('drop_out'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2_layer'):
    with tf.name_scope('weights'):
        W_fc2 = weight_variable([1024, 1])
    with tf.name_scope('biases'):
        b_fc2 = bias_variable([1])
    with tf.name_scope('output'):
        y_conv=tf.matmul(h_fc1_drop, W_fc2) + b_fc2

print "start session"
sess = tf.InteractiveSession()


with tf.name_scope('square'):
    with tf.name_scope('loss'):
        loss = tf.reduce_sum(tf.square(y_conv - y_))
tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


with tf.name_scope('pearson'):
    _, pearson = tf.contrib.metrics.streaming_pearson_correlation(y_conv, y_)
    sess.run(tf.local_variables_initializer())
tf.summary.scalar('pearson', pearson)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("logs/",sess.graph)
test_writer = tf.summary.FileWriter("logs/")

sess.run(tf.global_variables_initializer())
print "start import data"
STS_train = data_helper.dataset(s1=s1_train, s2=s2_train, label=label_train)
print "初始化完毕，开始训练"
for i in range(40000):
    batch_train = STS_train.next_batch(50)
    # 训练模型
    train_step.run(feed_dict={x1: batch_train[0], x2:batch_train[1], y_: batch_train[2], keep_prob: 0.5})
    # 对结果进行记录
    if i % 100 == 0:
        train_result = sess.run(merged, feed_dict={
            x1: batch_train[0], x2: batch_train[1], y_: batch_train[2], keep_prob: 1.0})
        train_writer.add_summary(train_result, i)
        train_pearson = pearson.eval(feed_dict={
            x1: batch_train[0], x2: batch_train[1], y_: batch_train[2], keep_prob: 1.0})
        train_loss = loss.eval(feed_dict={
            x1: batch_train[0], x2: batch_train[1], y_: batch_train[2], keep_prob: 1.0})

        print "step %d, training pearson %g, loss %g" % (i, train_pearson, train_loss)

# STS_test = data_helper.dataset(s1=s1_test, s2=s2_test, label=label_test)
print "test pearson %g"%pearson.eval(feed_dict={
    x1: s1_test, x2: s2_test, y_: label_test, keep_prob: 1.0})
print "test loss %g"%loss.eval(feed_dict={
    x1: s1_test, x2: s2_test, y_: label_test, keep_prob: 1.0})





