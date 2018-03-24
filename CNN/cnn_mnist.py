# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np

# 55000 * 28 * 28
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data",one_hot=True)

#
input_x = tf.placeholder(tf.float32, [None,28 * 28]) / 255
output_y = tf.placeholder(tf.int32,[None,10])

#
input_x_images = tf.reshape(input_x ,[-1,28,28,1])

#Test Set     3000
test_x = mnist.test.images[0:3000]
test_y = mnist.test.labels[0:3000]

# Build Network
# First Conv Layer
conv1 = tf.layers.conv2d(
                            inputs=input_x_images,
                            filters=32,
                            kernel_size=[5,5],
                            strides=1,
                            padding='same',
                            activation=tf.nn.relu
                        )
# Shaper 32 * 28 * 28

# First Pooling Layer
pool1 = tf.layers.max_pooling2d(
                                    inputs=conv1,
                                    pool_size=[2,2],
                                    strides=2
                                )
# Shaper [14,14,32]

# 2nd Conv Layer
conv2 = tf.layers.conv2d(
                            inputs=pool1,
                            filters=64,
                            kernel_size=[5,5],
                            strides=1,
                            padding='same',
                            activation=tf.nn.relu
                        )

# First Pooling Layer
pool2 = tf.layers.max_pooling2d(
                                    inputs=conv2,
                                    pool_size=[2,2],
                                    strides=2
                                )
# Shaper [7,7,64]

# flat
flat = tf.reshape(pool2, [-1,7 * 7 * 64])

# 1024
dense = tf.layers.dense(inputs=flat,units=1024,activation=tf.nn.relu)

# dropout 0.5
dropout = tf.layers.dropout(inputs=dense,rate=0.5,training=True)

logits = tf.layers.dense(inputs=dropout,units=10)

# Cross entropy And Softmax
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y,logits=logits)

# Adam
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

accuracy = tf.metrics.accuracy(
                                labels=tf.argmax(output_y,axis=1),
                                predictions=tf.argmax(logits,axis=1)
                                )[1]


session = tf.Session()

init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
session.run(init)


for i in range(20000):
    batch = mnist.train.next_batch(50)
    train_loss,train_op_ = session.run([loss,train_op],{input_x:batch[0],output_y:batch[1]})
    if i % 100 == 0:
        test_accuracy = session.run(accuracy,{input_x:test_x,output_y:test_y})
        print("Step=%d,Train loss=%.4f,[Test accuracy=%2f]" % (i,train_loss,test_accuracy))


test_output = session.run(logits,{input_x : test_x[0:20]})
inferenced_y = np.argmax(test_output,1)

print(inferenced_y,'Inferenced number')
print(np.argmax(test_y[0:20],1))

session.close()
