#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 18:17:36 2017

@author: amuhebwa

Notes : All the 100K images cause my machine to freeze.
I am choosing a very small subset of the images to design the algorithm
and then transfer it to a cluster with more computing power
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
import tensorflow as tf


# converts a single image
def to_grayscale(im, weights=np.c_[0.2989, 0.5870, 0.1140]):
    tile = np.tile(weights, reps=(im.shape[0],im.shape[1],1))
    return np.sum(tile * im, axis=2)


# load the clean csv file with corresponding labels
# replace it with wikipaintings_labels.csv
def load_labels():
    _labels = pd.read_csv('labels.csv')
    _labels = _labels.set_index(_labels['Id'])
    _labels.drop(['Id'], axis=1, inplace=True)
    return _labels


def centered_crop(img, new_height, new_width):
    width = np.size(img, 1)
    height = np.size(img, 0)
    left = int(np.ceil((width - new_width)/2.))
    top = int(np.ceil((height - new_height)/2.))
    right = int(width - np.floor((width - new_width) / 2))
    bottom = int(np.floor((height + new_height)/2.))
    c_img = img[top:bottom, left:right]
    return c_img


# helper function that calls most of the functions above
def convert_image(image_location):
    # image = Image.open(imageLocation)
    image = plt.imread(image_location)
    image = centered_crop(image, 128, 128)
    image = to_grayscale(image)
    image = np.asarray(image, dtype=np.float32)
    # flatten the image and reshape it
    # a 128 * 128 image becomes
    # 128 * 128 = 16384 or [16384, 1]
    if (image.shape[0] == 128) & (image.shape[1] == 128):
        image = image.flatten()
        # image = image.reshape(image.shape[0], 1)
        return image
    return None


# function to load images from the images folder.
# my machine is weak, so i am loading a very small set of images to test the idea.
def prepare_dataset():
    _labels = load_labels()
    dir_name = os.getcwd() + '/images/'
    images_list = os.listdir(dir_name)
    _dataset = []
    for index in range(len(images_list)):
        image_location = dir_name + images_list[index]
        image_id = images_list[index][:-4]
        image_label = _labels.ix[int(image_id)]['Label']
        image_array = convert_image(image_location)
        if image_array is not None:
            _dataset.append([image_array, image_label])
    return _dataset


dataset = prepare_dataset()
x_data = [v[0] for v in dataset]
y_labels = [v[1] for v in dataset]
# convert them to arrays
images = np.asarray(x_data)
labels = np.asarray(y_labels)
# labels = labels.reshape(labels.shape[0], 1)
images = images/255.0

no_classes = len(np.unique(labels))
print(np.size(images))
# hot-encode the labels
labels = pd.get_dummies(labels).values


# helper class and functions
class DataSet(object):
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], ("images.shape: %s labels.shape: %s" % (images.shape,labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(_images, _labels):
    class DataSets(object):
        pass
    data_sets = DataSets()
    data_sets.train = DataSet(_images, _labels)
    return data_sets


mydata = read_data_sets(images, labels)


# --- start designing the network---
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x_im, wghts):
    return tf.nn.conv2d(x_im, wghts, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x_im):
    return tf.nn.max_pool(x_im, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder("float", shape=[None, 16384])
y_ = tf.placeholder("float", shape=[None, no_classes])
x_image = tf.reshape(x, [-1, 128, 128, 1])

# --First layer
w_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# -- second layer ---
w_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# --- third layer
w_conv3 = weight_variable([3, 3, 64, 96])
b_conv3 = bias_variable([96])
h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# ---fourth layer ---
w_conv4 = weight_variable([3, 3, 96, 128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_pool3, w_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

# --- fifth layer ---
w_conv5 = weight_variable([3, 3, 128, 256])
b_conv5 = bias_variable([256])
h_conv5 = tf.nn.relu(conv2d(h_pool4, w_conv5) + b_conv5)
h_pool5 = max_pool_2x2(h_conv5)


W_fc1 = weight_variable([4 * 4 * 256, 1024])
b_fc1 = bias_variable([1024])

h_pool5_flat = tf.reshape(h_pool5, [-1, 4*4*256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, no_classes])
b_fc2 = bias_variable([no_classes])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 5
iterations = 10000
for i in range(iterations):
    batch = mydata.train.next_batch(batch_size)
    train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g" % (i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % sess.run(accuracy, feed_dict={
       x: mydata.train.images, y_: mydata.train.labels, keep_prob: 1.0}))
