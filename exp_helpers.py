from tensorflow.examples.tutorials.mnist import input_data
import datetime
import tensorflow as tf
import os
import copy
import math
import random
import sys
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

# facilitates the damaging of the network.
def damage_network(network_matrices, dmg_size, damage_amt):
    matrix_shapes = get_matrix_shapes(network_matrices)
    matrices_as_vector = vectorize_network(network_matrices)
    damage_indices = get_damage_indices(matrices_as_vector, dmg_size)
    matrices_as_vector[damage_indices] = damage_amt
    return reshape_matrices(matrices_as_vector, matrix_shapes)

# filter network:
# filter_type = "inside", filters inside-out.
# filter_type = "outside", filters outside-in.
# filters from median + and - the percentile window size, so the window is actually 2*percentile_window in size.
def filter_network(network_matrices, percentile_window, damage_amt, filter_type):
    matrix_shapes = get_matrix_shapes(network_matrices)
    matrices_as_vector = vectorize_network(network_matrices)
    if filter_type == "inside":
        return_vector = filter_vector_in(matrices_as_vector, percentile_window, damage_amt)
    elif filter_type == "outside":
        return_vector = filter_vector_out(matrices_as_vector, percentile_window, damage_amt)
    return reshape_matrices(return_vector, matrix_shapes)
    
# returns shapes of original network matrices for reshaping
def get_matrix_shapes(network_matrices):
    list_of_shapes = []
    for matrix in network_matrices:
        list_of_shapes.append(list(matrix.shape))
    return list_of_shapes

# turns network matrices into one long vector
def vectorize_network(network_matrices):
    vector = np.empty(0)
    for matrix in network_matrices:
        vector = np.append(vector, np.reshape(copy.copy(matrix), -1))
    return vector

# returns random sample of indices to damage
def get_damage_indices(matrices_as_vector, dmg_size):
    num_elements_to_damage = int(math.floor(dmg_size * len(matrices_as_vector)))
    non_zero_elements = np.nonzero(matrices_as_vector)
    linear_indices = random.sample(range(0, len(non_zero_elements[0])), num_elements_to_damage)
    return non_zero_elements[0][linear_indices]

# reshapes damaged vector into original network matrices
def reshape_matrices(matrix_as_vector, matrix_shapes):
    matrices = []
    vector_lengths = get_vector_lengths(matrix_shapes)
    for i in range(len(matrix_shapes)):
        matrices.append(\
              np.reshape(\
                matrix_as_vector[sum(vector_lengths[0:i+1]):sum(vector_lengths[0:(i+2)])],\
                       matrix_shapes[i]))
    return matrices

def get_vector_lengths(matrix_shapes):
    length = [0]
    for shape in matrix_shapes:
        length.append(np.prod(shape))
    return length

# helper function for filter, inside damage
def filter_vector_in(matrices_as_vector, percentile_window, damage_amt):
    upper_perc = np.percentile(matrices_as_vector, 50 + percentile_window)
    lower_perc = np.percentile(matrices_as_vector, 50 - percentile_window)
    for i in range(len(matrices_as_vector)):
        if (matrices_as_vector[i] <= upper_perc and matrices_as_vector[i] >= lower_perc):
            matrices_as_vector[i] = damage_amt
    return matrices_as_vector

# helper function for filter, outside damage
def filter_vector_out(matrices_as_vector, percentile_window, damage_amt):
    upper_perc = np.percentile(matrices_as_vector, 100 - percentile_window)
    lower_perc = np.percentile(matrices_as_vector, 0 + percentile_window)
    for i in range(len(matrices_as_vector)):
        if (matrices_as_vector[i] > upper_perc or matrices_as_vector[i] < lower_perc):
            matrices_as_vector[i] = damage_amt
    return matrices_as_vector
    
# returns final output values for every class by image.
def get_output_class_vectors(network_matrices, sess, y_conv, x, test_images, keep_prob, W_conv1, W_conv2, W_fc1):
    return sess.run(y_conv, feed_dict={x: test_images, 
                               keep_prob: 1.0,
                                 W_conv1: network_matrices[0],
                                 W_conv2: network_matrices[1],
                                   W_fc1: network_matrices[2]})

# returns labels predicted by the network
def get_predicted_labels(predicted_vectors):
    return np.argmax(predicted_vectors, axis=1)
    
# returns accuracy of the network
def get_network_accuracy(actual_labels, predicted_labels):
    errors = np.subtract(actual_labels, predicted_labels)
    errors[np.nonzero(errors)] = 1
    return 1 - float(sum(errors))/float(len(errors))

# handles printing everything to .csv file. 
def output_data_to_csv(file_name, damage_size, trial_number, actual_labels, predicted_labels, class_scores):
    indices = range(len(actual_labels))
    fd = open(file_name, 'a')
    is_wrong = 0
    for i in range(len(actual_labels)):
        if actual_labels[i] - predicted_labels[i] == 0:
            is_wrong = 0
        else:
            is_wrong = 1
        fd.write('%d,%f,%f,%d,%d,%d,' % (i, damage_size, trial_number, actual_labels[i], predicted_labels[i], is_wrong))
        for class_score in class_scores[i]:
            fd.write('%f,' % class_score) 
        fd.write('\n')
    fd.close
    
# returns new and unique file name
def get_file_name(trial_counter):
    return ("mnist_cnn_trial_%d_" % trial_counter) + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f.csv")

# creates a new .csv file in working directory
def initialize_new_file(header_string, trial_counter):
    file_name = get_file_name(trial_counter)
    fd = open(file_name, 'a')
    fd.write(header_string) 
    fd.close()
    return file_name

# Returns actual image labels
def get_actual_image_labels(sess, actual, y_, test_labels):
    return sess.run(actual, feed_dict={y_: test_labels})








































































































