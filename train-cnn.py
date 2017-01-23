import pickle
import pandas as pd
import math
import os
import numpy as np
from itertools import product
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.python.ops.variables import Variable
from tensorflow.contrib.layers import flatten

import helpers as h

#---------------------------------------------
# Training parameters
#---------------------------------------------
training_file = '/home/mattwg/Projects/carnd-project2/data/train.p'
testing_file = '/home/mattwg/Projects/carnd-project2/data/test.p'
features_count = 32 * 32
batch_size=100
epochs = 2000
learning_rate = 0.00001
early_stopping_rounds = 10
dropout_probability = 0.2
TRAIN_DIR = 'logs/'

#---------------------------------------------
# Load data
#---------------------------------------------
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(X_train)
n_test = len(X_test)
image_shape = X_train.shape[1]
n_classes = len(set(y_train))

#---------------------------------------------
# Preprocessing of images 
#---------------------------------------------

X_train_gray = np.empty( [n_train, image_shape, image_shape], dtype = np.int32)
for i, img in enumerate(X_train):
    X_train_gray[i,:,:] = cv2.equalizeHist(h.grayscale(img), (0, 254) )
X_test_gray = np.empty( [n_test, image_shape, image_shape], dtype = np.int32)
for i, img in enumerate(X_test):
    X_test_gray[i,:,:] = cv2.equalizeHist(h.grayscale(img), (0, 254) )

    
X_train_gray, X_valid_gray, y_train, y_valid = train_test_split( 
    X_train_gray,
    y_train,
    test_size=0.10,
    random_state=1973)

encoder = LabelBinarizer()
encoder.fit(y_train)
train_labels = encoder.transform(y_train)
valid_labels = encoder.transform(y_valid)
test_labels = encoder.transform(y_test)

# Change to float32 for Tensorflow
train_labels = train_labels.astype(np.float32)
valid_labels = valid_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

X_train_gray_flat = h.flatten_all_gray(X_train_gray)
X_valid_gray_flat = h.flatten_all_gray(X_valid_gray)
X_test_gray_flat = h.flatten_all_gray(X_test_gray)

X_train_gray_flat = h.normalize_grayscale(X_train_gray_flat)
X_valid_gray_flat = h.normalize_grayscale(X_valid_gray_flat)
X_test_gray_flat = h.normalize_grayscale(X_test_gray_flat)

#---------------------------------------------
# Convolutional Neural Net - LeNet
#---------------------------------------------

def LeNet2(x):
    x = tf.reshape(x, (-1, 32, 32, 1))
    # Convolution layer 1. The output shape should be 28x28x16.
    x=h.conv_layer(input=x, num_input_channels=1, filter_size=5, num_filters=16, stride=1, padding='VALID')
    x=tf.nn.relu(x)
    # Pooling layer 1. The output shape should be 14x14x16.
    x=tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Convolution layer 2. The output shape should be 10x10x32.
    x=h.conv_layer(input=x, num_input_channels=16, filter_size=5, num_filters=32, stride=1, padding='VALID')
    x=tf.nn.relu(x)
    # Pooling layer 2. The output shape should be 5x5x32.
    x=tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Flatten layer. Should have 5x5x32 = 800 outputs.
    x=tf.contrib.layers.flatten(x)
    # Fully connected layer 1. This should have 400 outputs.
    x=h.fully_connected_layer(x, 800, 400)
    x=tf.nn.relu(x)
    # Fully connected layer 2. This should have 100 outputs.
    x=h.fully_connected_layer(x, 400, 100)
    x=tf.nn.relu(x)
    # Fully connected layer 3. This should have 43 outputs.
    x=h.fully_connected_layer(x, 100, 43)
    # Return the result of the last fully connected layer.
    return x

#---------------------------------------------
# Construct Tensorflow Graph
#---------------------------------------------

# Important in Notebooks!
tf.reset_default_graph()

features_count = 32 * 32

features = tf.placeholder(tf.float32, [None, features_count])
labels = tf.placeholder(tf.float32, [None, n_classes]) 

# Note we reshape flattened feature back to 32x32 in LeNet:
leNet = LeNet2(features)

prediction = tf.nn.softmax(leNet)
predicted_class = tf.argmax(prediction, dimension=1)

#train_feed_dict = {features: X_train_gray_flat, labels: train_labels}
valid_feed_dict = {features: X_valid_gray_flat, labels: valid_labels}
#test_feed_dict = {features: X_test_gray_flat, labels: test_labels}

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(leNet, labels))

init = tf.initialize_all_variables()

#with tf.Session() as session:
    #session.run(init)
    #session.run(loss, feed_dict=train_feed_dict)
    #session.run(loss, feed_dict=valid_feed_dict)
    #session.run(loss, feed_dict=test_feed_dict)
    #biases_data = session.run(biases)
    
is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

X_train_gray_b = h.bin_data(X_train_gray, y_train)
g=h.data_generator(X_train_gray_b,1000)

result = h.train_model_generator(model_name='lenet2',
                            init=init,
                            loss=loss,
                            features=features,
                            labels=labels,
                            generator=g,
                            batches_per_epoch = 100,
                            valid_feed_dict=valid_feed_dict,
                            accuracy=accuracy,
                            predicted_class=predicted_class,
                            epochs=1,
                            learning_rate=0.001,
                            early_stopping_rounds = 10,
                            opt="GD")
serialize_training_data('models/lenet2', result)

