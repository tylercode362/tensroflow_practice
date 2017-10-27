#import need packages

import numpy as np
import random
import tensorflow as tf

# create train data
np.random.seed(3)
train_data = np.random.randn(1000, 8,4)
train_label = (np.absolute(train_data)).astype(int).max(axis = 1)
print(train_label.shape)
dev_data = np.random.randn(10,8,4)
dev_label = np.absolute(dev_data.copy()).astype(int).max(axis = 1)

print("train set input")
print(train_data[:10])
print("train set label")
print(train_label[:10])
print("dev set")
print(dev_data[:10])
print("dev set label")
print(dev_label[:10])

def model(w, b, kind, width, height):
    #1 layer hidden unit
    x = tf.placeholder(tf.float32, [None, width * height])
    x_image = tf.reshape(x, shape=[-1, height, width, 1])
    y_ = tf.placeholder(tf.float32, [None, kind])
    keep_prob = tf.placeholder(tf.float32)
    w_alpha = 0.00001
    b_alpha = 0.0001
    layers_dims = [1, 32]
    L = len(layers_dims)
    
    input_layer_data = tf.reshape(x_image, [-1, height, width, 1])
    new_width, new_height = width, height
    for l in range(1,L):
        input_layer_data = tf.reshape(input_layer_data, [-1, height, width, 1])
        w = tf.Variable(w_alpha*tf.random_normal([3, 3, layers_dims[l-1], layers_dims[l]])) 
        b = tf.Variable(b_alpha*tf.random_normal([layers_dims[l]]))
        conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_layer_data, w, strides=[1, 1, 1, 1], padding='SAME'), b))
        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.dropout(conv, keep_prob)
        input_layer_data = conv
        new_width = int(new_width / 2)
        new_height = int(new_height / 2) 

        
    w_d = tf.Variable(w_alpha*tf.random_normal([int(new_width * new_height * layers_dims[l]), 256]))
    b_d = tf.Variable(b_alpha*tf.random_normal([256]))
    dense = tf.reshape(conv, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out1 = tf.Variable(w_alpha*tf.random_normal([256, 1]))
    b_out1 = tf.Variable(b_alpha*tf.random_normal([1]))
    w_out2 = tf.Variable(w_alpha*tf.random_normal([256, 1]))
    b_out2 = tf.Variable(b_alpha*tf.random_normal([1]))
    w_out3 = tf.Variable(w_alpha*tf.random_normal([256, 1]))
    b_out3 = tf.Variable(b_alpha*tf.random_normal([1]))
    w_out4 = tf.Variable(w_alpha*tf.random_normal([256, 1]))
    b_out4 = tf.Variable(b_alpha*tf.random_normal([1]))

    y_conv1 = tf.add(tf.matmul(dense, w_out1), b_out1)    
    y_conv2 = tf.add(tf.matmul(dense, w_out2), b_out2)    
    y_conv3 = tf.add(tf.matmul(dense, w_out3), b_out3)    
    y_conv4 = tf.add(tf.matmul(dense, w_out4), b_out4)
    out = tf.concat([y_conv1, y_conv2, y_conv3, y_conv3],1, name="out")
    return x, keep_prob, y_, out

width = train_data.shape[2]
height = train_data.shape[1]
train_input_data = train_data.reshape((train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
dev_input_data = dev_data.reshape((dev_data.shape[0], dev_data.shape[1]*dev_data.shape[2]))
x, keep_prob, y_, out = model(0.001, 0.001, 4, width, height)

#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_group, labels=y_))
with tf.name_scope('CrossEntropy'):
    loss = tf.reduce_mean(tf.pow((out - y_),2))
    #loss = tf.reduce_mean(tf.pow(out - y_))

lost_wist = tf.summary.scalar("Loss", loss)
    
with tf.name_scope('GradientDescent'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
#get closest int
out_rint = tf.cast(tf.rint(out), tf.int64)
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.cast(y_, tf.int64), out_rint)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_wist = tf.summary.scalar('Accuracy', accuracy)

print("start")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('logs/train/',  sess.graph)
    dev_writer = tf.summary.FileWriter('logs/dev/',  sess.graph)
    
    try :
        for i in range(100000) :
            _, loss_ = sess.run([optimizer, loss], feed_dict={ x: train_input_data, y_: train_label, keep_prob: 0.5}) 

            if(i % 300 == 0) :
                train_summary, train_accuracy_result, train_loss = sess.run([summary, accuracy, loss] , feed_dict={ x: train_input_data , y_: train_label , keep_prob: 1})
                dev_summary, dev_accuracy_result, dev_loss = sess.run([summary, accuracy, loss] , feed_dict={ x: dev_input_data , y_: dev_label , keep_prob: 1}) 
                train_writer.add_summary(train_summary, i)
                train_writer.flush()
                dev_writer.add_summary(dev_summary, i)
                dev_writer.flush()
                print("step - "+ str(i), " train loss is ", train_loss)
                print("train accuracy", train_accuracy_result)
                print("dev accuracy", dev_accuracy_result)
    except KeyboardInterrupt :
        train_writer.close()
        dev_writer.close()
        sess.close()
    train_writer.close()
    dev_writer.close()