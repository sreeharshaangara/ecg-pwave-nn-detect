from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as krs
from tensorflow.keras.layers import Dense, Dropout


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import wfdb
import wfdb.io as wfdbio

model = krs.models.load_model('current_run_CNN.h5')


INPUT_FILE_PATH = '..\\..\\mit-bih-database\\103'

DATA_LIMIT = 20000
NN_INPUT_DATA_LENGTH = 320

num_training_sets  = int( DATA_LIMIT / NN_INPUT_DATA_LENGTH)

# Read input data
record = wfdbio.rdsamp(INPUT_FILE_PATH, channels = [0])
input_dat = record[0][:DATA_LIMIT].squeeze().tolist()

# Read Annotations
annotation = wfdb.rdann(INPUT_FILE_PATH, 'pwave')
annotation_timed = []
ann_ptr = 0

# Organize annotation to array
for i in range(0, DATA_LIMIT):
    if(ann_ptr < annotation.sample.size):
        if(i == annotation.sample[ann_ptr]):
            annotation_timed.append(1)
            ann_ptr += 1
        else:
            annotation_timed.append(0) 
    else:
        annotation_timed.append(0)

############################
# Form input data into sets
############################
norm_input_dat = []
for i in range(0, num_training_sets):
    #Normalize input
    tmp_dat_max = max(input_dat[i*NN_INPUT_DATA_LENGTH:(i+1)*NN_INPUT_DATA_LENGTH])
    tmp_dat_min = min(input_dat[i*NN_INPUT_DATA_LENGTH:(i+1)*NN_INPUT_DATA_LENGTH])
    norm_dat = []
    for j in range (0, NN_INPUT_DATA_LENGTH):
        norm_dat.append((input_dat[i*NN_INPUT_DATA_LENGTH + j] - tmp_dat_min)/(tmp_dat_max - tmp_dat_min))

    norm_input_dat.append(norm_dat)

norm_input_dat = np.array(norm_input_dat, dtype=np.float32)

# norm_input_dat = []
# for i in range(0, (DATA_LIMIT-NN_INPUT_DATA_LENGTH)):
#     tmp_dat_max = max(input_dat[i:(i+NN_INPUT_DATA_LENGTH)])
#     tmp_dat_min = min(input_dat[i:(i+NN_INPUT_DATA_LENGTH)])
#     norm_dat = []
    
#     for j in range (0, NN_INPUT_DATA_LENGTH):
#         norm_dat.append((input_dat[i + j] - tmp_dat_min)/(tmp_dat_max - tmp_dat_min))

#     norm_input_dat.append(norm_dat)

# norm_input_dat = np.array(norm_input_dat)

predicted_out = (model.predict(norm_input_dat, batch_size = 1))

print(predicted_out)

#Reform data into graph
predicted_out_graph = []

for i in range(0, num_training_sets):
    for j in range(0, NN_INPUT_DATA_LENGTH):
        predicted_out_graph.append(0)
    
    if(predicted_out[i][0] > 0):
        predicted_out_graph[(i*NN_INPUT_DATA_LENGTH) + int(predicted_out[i][0] * NN_INPUT_DATA_LENGTH)] = 1.5
        if(predicted_out[i][1] > 0):
            print("Second peak")
            predicted_out_graph[(i*NN_INPUT_DATA_LENGTH) + int(predicted_out[i][1] * NN_INPUT_DATA_LENGTH)] = 1.5


plot_size = num_training_sets*NN_INPUT_DATA_LENGTH

t = range(0,plot_size)

plt.plot(t, annotation_timed[:plot_size], t, input_dat[:plot_size], t , predicted_out_graph)

plt.show()


