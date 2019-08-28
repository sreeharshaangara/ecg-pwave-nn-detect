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


INPUT_FILE_PATH = ['..\\..\\mit-bih-database\\100', '..\\..\\mit-bih-database\\101']
NN_INPUT_DATA_LENGTH = 320
TRAINING_DATA_LIMIT = 80000

num_training_sets = int(TRAINING_DATA_LIMIT/NN_INPUT_DATA_LENGTH)

def Read_Data_from_wfd(filepath):
    # Read input data
    record = wfdbio.rdsamp(filepath, channels = [0])
    input_dat = record[0].squeeze().tolist()

    # Read Annotations
    annotation = wfdb.rdann(filepath, 'pwave')

    return(input_dat, annotation)


annotation_pointers_pos = []
annotation_pointers_neg = []
ann_ptr = 0

# Pre-work annotation to array
for i in range(0, num_training_sets):

    # Check for first peak
    if(i*NN_INPUT_DATA_LENGTH <= annotation.sample[ann_ptr] < (i+1)*NN_INPUT_DATA_LENGTH):
        annotation_pointers_pos.append(annotation.sample[ann_ptr] - i*NN_INPUT_DATA_LENGTH)
        ann_ptr += 1
        # Check for second peak
        if(i*NN_INPUT_DATA_LENGTH <= annotation.sample[ann_ptr] < (i+1)*NN_INPUT_DATA_LENGTH):
            annotation_pointers_neg.append(annotation.sample[ann_ptr] - i*NN_INPUT_DATA_LENGTH)
            ann_ptr += 1
        else:
            annotation_pointers_neg.append(0) 

    else:
        annotation_pointers_pos.append(0) 
        annotation_pointers_neg.append(0) 

#print(annotation_pointers_pos)
#print(annotation_pointers_neg)
#training_output_dat = np.array(annotation_pointers_pos.append(annotation_pointers_neg)) / NN_INPUT_DATA_LENGTH


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
#training_output_dat = np.array(annotation_pointers) / NN_INPUT_DATA_LENGTH
training_output_dat = [annotation_pointers_pos, annotation_pointers_neg]
training_output_dat = np.array(training_output_dat) / NN_INPUT_DATA_LENGTH

training_output_dat = np.swapaxes(training_output_dat,0,1)
print(training_output_dat) 
print(training_output_dat.shape) 

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
model.add(Dense(128, input_dim=NN_INPUT_DATA_LENGTH, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='relu'))
model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(training_input_dat, training_output_dat,
          epochs=1000, shuffle = True,
          batch_size=NN_INPUT_DATA_LENGTH)

model.save('current_run.h5')          

#score = model.evaluate(test_input_dat, test_output_dat)

predicted_out = model.predict(training_input_dat)

print(predicted_out)
print(training_output_dat)

# for i in range(0, predicted_out.size):
#     if(predicted_out[i] > 0.5):
#         print("yay")




