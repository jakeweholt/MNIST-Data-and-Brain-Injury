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
def get_output_class_vectors(network_matrices):
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
def get_file_name():
    return ("mnist_cnn_trial_%d_" % trial_counter) + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f.csv")

# creates a new .csv file in working directory
def initialize_new_file():
    file_name = get_file_name()
    fd = open(file_name, 'a')
    fd.write(header_string) 
    fd.close()
    return file_name

# Returns actual image labels
def get_actual_image_labels():
    return sess.run(actual, feed_dict={y_: test_labels})


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

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 10])
b_fc1 = bias_variable([10])

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
actual_test_image_labels = get_actual_image_labels()
trial_counter = 1

############
# Damage and file output loop:
while True:
    file_name = initialize_new_file()
    for dmg_size in damages_values:
        damaged_network = damage_network(matrices_to_damage, dmg_size, default_damage_amount)
        predicted_vectors = get_output_class_vectors(damaged_network)
        predicted_test_image_labels = get_predicted_labels(predicted_vectors)
        network_accuracy = get_network_accuracy(actual_test_image_labels, predicted_test_image_labels)
        output_data_to_csv(file_name, dmg_size, trial_counter, actual_test_image_labels, predicted_test_image_labels, predicted_vectors)
    print("Trials completed: %d\n" % trial_counter)
    trial_counter = trial_counter + 1