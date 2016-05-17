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
default_damage_amount = 0 # CHANGE
damages_values = np.arange(0,1,0.01) # CHANGE 
print(damages_values)

############
# Output file header for the top of the csv file.
header_string = "image_index, damage_size, trial, correct_class, inferred_class, is_wrong, pred_0, pred_1" +\
                ", pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9\n" # CHANGE

############
[test_images, test_labels] = h.prepare_data()
[sess, actual, y_, y_conv, x, keep_prob, W_conv1, W_conv2, W_fc1, matrices_to_damage] = h.setup_network()


############ 
# List of actual test image labels for comparison.
actual_test_image_labels = h.get_actual_image_labels(sess, actual, y_, test_labels)
trial_counter = 1

############
# Damage and file output loop:
accuracies = np.zeros((len(damages_values), 3))
while True:
    file_name = h.initialize_new_file(header_string, trial_counter)
    dmg_counter = 0;
    for dmg_size in damages_values:
        [damaged_network, num_damaged] = h.damage_network(matrices_to_damage, dmg_size, default_damage_amount)
        predicted_vectors = h.get_output_class_vectors(damaged_network, sess, y_conv, x, test_images, keep_prob, W_conv1, W_conv2, W_fc1)
        predicted_test_image_labels = h.get_predicted_labels(predicted_vectors)
        network_accuracy = h.get_network_accuracy(actual_test_image_labels, predicted_test_image_labels)
        accuracies[dmg_counter, 0] = dmg_size
        accuracies[dmg_counter, 1] = num_damaged
        accuracies[dmg_counter, 2] = network_accuracy
        dmg_counter = dmg_counter + 1;
    h.output_summary_data_to_csv(file_name, accuracies, trial_counter)
    print("Trials completed: %d\n" % trial_counter)
    trial_counter = trial_counter + 1