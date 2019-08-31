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
TRAINING_DATA_LIMIT = 300000

num_training_sets = int(TRAINING_DATA_LIMIT/NN_INPUT_DATA_LENGTH)

def Read_Data_from_wfd(filepath):
    # Read input data
    record = wfdbio.rdsamp(filepath, channels = [0])
    input_dat = record[0].squeeze().tolist()

    # Read Annotations
    annotation = wfdb.rdann(filepath, 'pwave')

    return(input_dat, annotation)

def pre_process_data(input_dat, annotation):
    
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

    training_output_dat = [annotation_pointers_pos, annotation_pointers_neg]
    training_output_dat = np.array(training_output_dat) / NN_INPUT_DATA_LENGTH

    training_output_dat = np.swapaxes(training_output_dat,0,1)

    return(training_input_dat, training_output_dat)


def plot_matplotlib_model(history):

    history_dict = history.history
    # print(history_dict.keys())

    # sys.exit()


    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show(block = False)

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()



#########
# Main
#########
data_file, annotation_file = Read_Data_from_wfd(INPUT_FILE_PATH[0])
input_dat_1, output_dat_1 = pre_process_data(data_file, annotation_file)
data_file, annotation_file = Read_Data_from_wfd(INPUT_FILE_PATH[1])
input_dat_2, output_dat_2 = pre_process_data(data_file, annotation_file)

training_input_dat = np.concatenate((input_dat_1, input_dat_2), axis=0)
training_output_dat = np.concatenate((output_dat_1, output_dat_2), axis=0)

#print(training_input_dat.shape)

#sys.exit()
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
model.add(Dense(32, input_dim=NN_INPUT_DATA_LENGTH, activation='relu'))
model.add(Dense(32, activation='linear'))
#model.add(Dense(64, activation='sigmoid'))
#model.add(Dense(128, activation='sigmoid'))
model.add(Dense(2, activation='relu'))
model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['accuracy'])


history = model.fit(training_input_dat, training_output_dat,
          epochs=500, validation_split=0.1,
          batch_size=16)

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model)
  

model.save('current_run.h5')          


#score = model.evaluate(test_input_dat, test_output_dat)

predicted_out = model.predict(training_input_dat)

print(predicted_out)
print(training_output_dat)

plot_matplotlib_model(history)

# for i in range(0, predicted_out.size):
#     if(predicted_out[i] > 0.5):
#         print("yay")




