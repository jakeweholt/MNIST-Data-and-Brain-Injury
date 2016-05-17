from tensorflow.examples.tutorials.mnist import input_data
import datetime
import tensorflow as tf
import os
import copy
import math
import random
import sys
import numpy as np
import exp_helpers as h

############
# User defined model parameters:
# default_damage_amount = what the weights are set to when they are damaged
# damages_values = range of values, 0 to 1 in steps of 0.01 to represent network damage amount.
default_damage_amount = 0
damages_values = np.arange(0,1,0.01) 
print(damages_values)

############
# Output file header for the top of the csv file.
header_string = "image_index, damage_size, trial, correct_class, inferred_class, is_wrong, pred_0, pred_1" +\
                ", pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9\n"

############
# Create TensorFlow data objects (contains all images and labels)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

############
# Even out class sizes, 892 images each, limited by the smallest class of only 892 images.
test_images = []
test_labels = []
counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
magic_val = 852
count = 0
for i in range(len(mnist.test.labels)):
    index = np.nonzero(mnist.test.labels[i])[0][0]
    if counts[index] < magic_val:
        test_images.append(mnist.test.images[i])
        test_labels.append(mnist.test.labels[i])
        count = count + 1
    counts[index] = counts[index] + 1

# New evenly sized test sets.
test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)

############
# TensorFlow setup
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

W_conv1 = h.weight_variable([5, 5, 1, 32])
b_conv1 = h.bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(h.conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = h.max_pool_2x2(h_conv1)

W_conv2 = h.weight_variable([5, 5, 32, 64])
b_conv2 = h.bias_variable([64])

h_conv2 = tf.nn.relu(h.conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = h.max_pool_2x2(h_conv2)

W_fc1 = h.weight_variable([7 * 7 * 64, 10])
b_fc1 = h.bias_variable([10])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
keep_prob = tf.placeholder("float")
y_conv = tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

predicted = tf.argmax(y_conv, 1)
actual = tf.argmax(y_, 1)
correct_prediction = tf.equal(predicted, actual)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

############
# import already trained model/network from file.
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(os.getcwd())
saver.restore(sess, ckpt.model_checkpoint_path)

############
# Convert weight matrices from tensorflow tensor objects to real value numpy arrays with sess.run()
# packs them together in a list to contain all of the weight matrices.
matrices_to_damage =\
    [np.asarray(sess.run(W_conv1)),
     np.asarray(sess.run(W_conv2)),
     np.asarray(sess.run(W_fc1))]
    

############ 
# List of actual test image labels for comparison.
actual_test_image_labels = h.get_actual_image_labels(sess, actual, y_, test_labels)
trial_counter = 1

############
# Damage and file output loop:
while True:
    file_name = h.initialize_new_file(header_string, trial_counter)
    for dmg_size in damages_values:
        damaged_network = h.damage_network(matrices_to_damage, dmg_size, default_damage_amount)
        predicted_vectors = h.get_output_class_vectors(damaged_network, sess, y_conv, x, test_images, keep_prob, W_conv1, W_conv2, W_fc1)
        predicted_test_image_labels = h.get_predicted_labels(predicted_vectors)
        network_accuracy = h.get_network_accuracy(actual_test_image_labels, predicted_test_image_labels)
        h.output_data_to_csv(file_name, dmg_size, trial_counter, actual_test_image_labels, predicted_test_image_labels, predicted_vectors)
    print("Trials completed: %d\n" % trial_counter)
    trial_counter = trial_counter + 1