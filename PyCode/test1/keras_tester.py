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

model = krs.models.load_model('current_run.h5')


INPUT_FILE_PATH = '..\\..\\mit-bih-database\\100'

DATA_LIMIT = 10000

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

predicted_out = model.predict(input_dat, batch_size = 1)

print(predicted_out)

t = range(0,DATA_LIMIT)

plt.plot(t, annotation_timed, t, input_dat, t , predicted_out)

plt.show()


