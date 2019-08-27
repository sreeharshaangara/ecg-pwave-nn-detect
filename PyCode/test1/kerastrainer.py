from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import wfdb
import wfdb.io as wfdbio


NN_INPUT_DATA_LENGTH = 350

INPUT_FILE_PATH = '..\\..\\mit-bih-database\\100'

TRAINING_DATA_LIMIT = 40000

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

# Form input data into sets
training_input_dat = []
for i in range(0, (TRAINING_DATA_LIMIT-NN_INPUT_DATA_LENGTH)):
    #training_input_dat.append((input_dat[i:(i+1)*NN_INPUT_DATA_LENGTH]))
    #training_input_dat.append(np.array(input_dat[i*NN_INPUT_DATA_LENGTH:(i+1)*NN_INPUT_DATA_LENGTH]))
    #training_input_dat.append(tf.convert_to_tensor(input_dat[i:(i+1)*NN_INPUT_DATA_LENGTH]))
    training_input_dat.append(np.array(input_dat[i:(i+NN_INPUT_DATA_LENGTH)]))

training_input_dat = np.array(training_input_dat)

#training_input_dat = np.array([np.array(xi) for xi in training_input_dat])


# Form Output data into sets
# training_output_dat = []
# for i in range(0, (TRAINING_DATA_LIMIT % NN_INPUT_DATA_LENGTH)):
#     #training_output_dat.append((annotation_timed[i:(i+1)*NN_INPUT_DATA_LENGTH]))
#     training_output_dat.append(np.array(annotation_timed[i*NN_INPUT_DATA_LENGTH:(i+1)*NN_INPUT_DATA_LENGTH]))
#     #training_output_dat.append(tf.convert_to_tensor(annotation_timed[i:(i+1)*NN_INPUT_DATA_LENGTH]))

# training_output_dat = np.array(training_output_dat)
training_output_dat = np.array(annotation_timed[NN_INPUT_DATA_LENGTH:TRAINING_DATA_LIMIT])

for i in range(0, training_output_dat.size):
    if(training_output_dat[i] > 0.5):
        print("yay")

print(training_input_dat)
print(training_output_dat)

print(training_input_dat.shape)
print(training_output_dat.shape)


# Create model
model = tf.keras.Sequential()



model.add(Dense(64, input_dim=NN_INPUT_DATA_LENGTH, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(training_input_dat, training_output_dat,
          epochs=5,
          batch_size=10)





model.save('current_run.h5')          

#score = model.evaluate(test_input_dat, test_output_dat)

predicted_out = model.predict(training_input_dat)

print(predicted_out)

for i in range(0, predicted_out.size):
    if(predicted_out[i] > 0.5):
        print("yay")





