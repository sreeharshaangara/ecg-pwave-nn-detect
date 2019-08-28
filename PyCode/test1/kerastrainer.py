from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import wfdb
import wfdb.io as wfdbio


NN_INPUT_DATA_LENGTH = 128

INPUT_FILE_PATH = '..\\..\\mit-bih-database\\100'

TRAINING_DATA_LIMIT = 20000

# Read input data
record = wfdbio.rdsamp(INPUT_FILE_PATH, channels = [0])
input_dat = record[0].squeeze().tolist()

# Read Annotations
annotation = wfdb.rdann(INPUT_FILE_PATH, 'pwave')
annotation_timed = []
ann_ptr = 0

# Pre-work annotation to array
for i in range(0, (TRAINING_DATA_LIMIT)):
    if(ann_ptr < annotation.sample.size):
        if(i == annotation.sample[ann_ptr]):
            annotation_timed.append(1.0)
            ann_ptr += 1
        else:
            annotation_timed.append(0.0) 
    else:
        annotation_timed.append(0.0)

############################
# Form input data into sets
############################
training_input_dat = []
for i in range(0, (TRAINING_DATA_LIMIT-NN_INPUT_DATA_LENGTH)):
    #training_input_dat.append((input_dat[i:(i+1)*NN_INPUT_DATA_LENGTH]))
    #training_input_dat.append(np.array(input_dat[i*NN_INPUT_DATA_LENGTH:(i+1)*NN_INPUT_DATA_LENGTH]))
    #training_input_dat.append(tf.convert_to_tensor(input_dat[i:(i+1)*NN_INPUT_DATA_LENGTH]))
    # training_input_dat.append(np.array(input_dat[i:(i+NN_INPUT_DATA_LENGTH)]))
    
    tmp_dat_max = max(input_dat[i:(i+NN_INPUT_DATA_LENGTH)])
    tmp_dat_min = min(input_dat[i:(i+NN_INPUT_DATA_LENGTH)])
    norm_dat = []
    
    for j in range (0, NN_INPUT_DATA_LENGTH):
        norm_dat.append((input_dat[i + j] - tmp_dat_min)/(tmp_dat_max - tmp_dat_min))
    training_input_dat.append(norm_dat)



training_input_dat = np.array(training_input_dat, dtype=np.float32)

print(training_input_dat.shape)


#training_input_dat = np.array([np.array(xi) for xi in training_input_dat])


############################
# Form Output data into sets
############################
# training_output_dat = []
# for i in range(0, (TRAINING_DATA_LIMIT % NN_INPUT_DATA_LENGTH)):
#     #training_output_dat.append((annotation_timed[i:(i+1)*NN_INPUT_DATA_LENGTH]))
#     training_output_dat.append(np.array(annotation_timed[i*NN_INPUT_DATA_LENGTH:(i+1)*NN_INPUT_DATA_LENGTH]))
#     #training_output_dat.append(tf.convert_to_tensor(annotation_timed[i:(i+1)*NN_INPUT_DATA_LENGTH]))

# training_output_dat = np.array(training_output_dat)
training_output_dat = np.array(annotation_timed[NN_INPUT_DATA_LENGTH:TRAINING_DATA_LIMIT])

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
# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.add(Dense(128, input_dim=NN_INPUT_DATA_LENGTH, activation='relu'))
# #model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# #model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])


#############
# 1D CNN 
#############
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(NN_INPUT_DATA_LENGTH,1)))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=1))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	
training_input_dat = np.reshape(training_input_dat, [-1, 128,1])

model.fit(training_input_dat, training_output_dat,
          epochs=5, validation_split = 0.2, shuffle = True,
          batch_size=NN_INPUT_DATA_LENGTH)


model.save('current_run.h5')          

#score = model.evaluate(test_input_dat, test_output_dat)

predicted_out = model.predict(training_input_dat)

print(predicted_out)

for i in range(0, predicted_out.size):
    if(predicted_out[i] > 0.5):
        print("yay")




