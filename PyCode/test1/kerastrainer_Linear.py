from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sys
import wfdb
import wfdb.io as wfdbio


NN_INPUT_DATA_LENGTH = 128
INPUT_FILE_PATH = '..\\..\\mit-bih-database\\100'
TRAINING_DATA_LIMIT = 20000

num_training_sets = int(TRAINING_DATA_LIMIT/NN_INPUT_DATA_LENGTH)

# Read input data
record = wfdbio.rdsamp(INPUT_FILE_PATH, channels = [0])
input_dat = record[0].squeeze().tolist()

# Read Annotations
annotation = wfdb.rdann(INPUT_FILE_PATH, 'pwave')
annotation_pointers = []
ann_ptr = 0

# Pre-work annotation to array
for i in range(0, num_training_sets):

    if(i*NN_INPUT_DATA_LENGTH <= annotation.sample[ann_ptr] <= (i+1)*NN_INPUT_DATA_LENGTH):
        annotation_pointers.append(annotation.sample[ann_ptr])
        ann_ptr += 1
    else:
        annotation_pointers.append(0.0) 

print(annotation_pointers)


############################
# Form input data into sets
############################
training_input_dat = []
for i in range(0, num_training_sets):
    #Normalize input
    tmp_dat_max = max(input_dat[i*NN_INPUT_DATA_LENGTH:(i+1)*NN_INPUT_DATA_LENGTH])
    tmp_dat_min = min(input_dat[i*NN_INPUT_DATA_LENGTH:(i+1)*NN_INPUT_DATA_LENGTH])
    norm_dat = []
    for j in range (0, NN_INPUT_DATA_LENGTH):
        norm_dat.append((input_dat[i*NN_INPUT_DATA_LENGTH + j] - tmp_dat_min)/(tmp_dat_max - tmp_dat_min))

    training_input_dat.append(norm_dat)

training_input_dat = np.array(training_input_dat, dtype=np.float32)


############################
# Form Output data into sets
############################
# training_output_dat = []
# for i in range(0, (TRAINING_DATA_LIMIT % NN_INPUT_DATA_LENGTH)):
#     #training_output_dat.append((annotation_timed[i:(i+1)*NN_INPUT_DATA_LENGTH]))
#     training_output_dat.append(np.array(annotation_timed[i*NN_INPUT_DATA_LENGTH:(i+1)*NN_INPUT_DATA_LENGTH]))
#     #training_output_dat.append(tf.convert_to_tensor(annotation_timed[i:(i+1)*NN_INPUT_DATA_LENGTH]))

# training_output_dat = np.array(training_output_dat)
training_output_dat = np.array(annotation_pointers)

# ############################
# # Form test data
# ############################
# test_input_dat = []
# for i in range(0, (TRAINING_DATA_LIMIT-NN_INPUT_DATA_LENGTH)):
#     test_input_dat.append(np.array(input_dat[i:(i+NN_INPUT_DATA_LENGTH)]))


# # for i in range(0, training_output_dat.size):
# #     if(training_output_dat[i] > 0.5):
# #         print("yay")

# print(training_input_dat)
# print(training_output_dat)

# print(training_input_dat.shape)
# print(training_output_dat.shape)


# Create model
model = tf.keras.Sequential()



##########
# MLP
##########
model.add(Dense(128, input_dim=NN_INPUT_DATA_LENGTH, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(training_input_dat, training_output_dat,
          epochs=5000, 
          batch_size=NN_INPUT_DATA_LENGTH)

model.save('current_run.h5')          

#score = model.evaluate(test_input_dat, test_output_dat)

predicted_out = model.predict(training_input_dat)

print(predicted_out)

# for i in range(0, predicted_out.size):
#     if(predicted_out[i] > 0.5):
#         print("yay")




