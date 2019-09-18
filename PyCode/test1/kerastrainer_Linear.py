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


INPUT_FILE_PATH = [ '..\\..\\mit-bih-database\\103',\
                    '..\\..\\mit-bih-database\\106',\
                    '..\\..\\mit-bih-database\\117',\
                    '..\\..\\mit-bih-database\\119',\
                    '..\\..\\mit-bih-database\\122',\
                    '..\\..\\mit-bih-database\\214',\
                    '..\\..\\mit-bih-database\\222',\
                    '..\\..\\mit-bih-database\\223',\
                    '..\\..\\mit-bih-database\\231']
# 320 samples(1 sec) windows
NN_INPUT_DATA_LENGTH = 320
# 25 mins of data from each
TRAINING_DATA_LIMIT = 480000

num_training_sets = int(TRAINING_DATA_LIMIT/NN_INPUT_DATA_LENGTH)

def Read_Data_from_wfd(filepath):
    # Read input data
    record = wfdbio.rdsamp(filepath, channels = [0])
    input_dat = record[0].squeeze().tolist()

    # Read Annotations
    annotation = wfdb.rdann(filepath, 'pwave')

    return(input_dat, annotation)

def pre_process_data(input_dat, annotation):
    
    ########################################################
    # Form input data and Output data into training sets
    ########################################################
    training_input_dat = []
    annotation_pointers_pos = []
    annotation_pointers_neg = []
    ann_ptr = 0

    for subchunk in range(0, num_training_sets):
        
        ## Optional, add some noise 
        # #Add some random noise/pure-noise sets        
        # noise_blank_out_decider = np.random.rand()

        # #Add noise to 20% of data
        # if(0.6 > noise_blank_out_decider > 0.80):
        #     noise_add_flag = 1
        # else:
        #     noise_add_flag = 0

        # #Pure noise to 10% of data
        # if(0.5 > noise_blank_out_decider > 0.6):
        #     pure_noise_flag = 1
        # else:
        #     pure_noise_flag = 0

        noise_add_flag = 0
        pure_noise_flag = 0

        # Make invalid ECG(pure noise) 
        if(pure_noise_flag):
            for index in range (0, NN_INPUT_DATA_LENGTH):
                input_dat[subchunk*NN_INPUT_DATA_LENGTH + index]  = np.random.rand() 
        else:
            if(noise_add_flag):
                #Get min/max to add relative noise
                tmp_dat_max = max(input_dat[subchunk*NN_INPUT_DATA_LENGTH:(subchunk+1)*NN_INPUT_DATA_LENGTH])
                tmp_dat_min = min(input_dat[subchunk*NN_INPUT_DATA_LENGTH:(subchunk+1)*NN_INPUT_DATA_LENGTH])

                # Add some noise, 10% of max-min
                for index in range (0, NN_INPUT_DATA_LENGTH):
                    input_dat[subchunk*NN_INPUT_DATA_LENGTH + index]  += 0.2 * np.random.rand() *(tmp_dat_max - tmp_dat_min)  


        #Normalize input
        norm_dat = []
        #Calc min/max for nomralizaton
        tmp_dat_max = max(input_dat[subchunk*NN_INPUT_DATA_LENGTH:(subchunk+1)*NN_INPUT_DATA_LENGTH])
        tmp_dat_min = min(input_dat[subchunk*NN_INPUT_DATA_LENGTH:(subchunk+1)*NN_INPUT_DATA_LENGTH])
        for index in range (0, NN_INPUT_DATA_LENGTH):
            norm_dat.append((input_dat[subchunk*NN_INPUT_DATA_LENGTH + index] - tmp_dat_min)/(tmp_dat_max - tmp_dat_min))

        training_input_dat.append(norm_dat)

        # Output 
        # Check for first peak
        if(subchunk*NN_INPUT_DATA_LENGTH <= annotation.sample[ann_ptr] < (subchunk+1)*NN_INPUT_DATA_LENGTH):
            # If pure noise, then just ignore any annotated peaks
            if(pure_noise_flag):
                annotation_pointers_pos.append(0)
            else:
                annotation_pointers_pos.append(annotation.sample[ann_ptr] - subchunk*NN_INPUT_DATA_LENGTH)
            ann_ptr += 1
            # Check for second peak
            if(subchunk*NN_INPUT_DATA_LENGTH <= annotation.sample[ann_ptr] < (subchunk+1)*NN_INPUT_DATA_LENGTH):
                # If pure noise, then just ignore any annotated peaks
                if(pure_noise_flag):
                    annotation_pointers_neg.append(0)
                else:
                    annotation_pointers_neg.append(annotation.sample[ann_ptr] - subchunk*NN_INPUT_DATA_LENGTH)
                ann_ptr += 1
            else:
                annotation_pointers_neg.append(0) 

        else:
            annotation_pointers_pos.append(0) 
            annotation_pointers_neg.append(0) 


    training_input_dat = np.array(training_input_dat, dtype=np.float32)


    # # Pre-work annotation to array
    # for i in range(0, num_training_sets):

    #     # Check for first peak
    #     if(i*NN_INPUT_DATA_LENGTH <= annotation.sample[ann_ptr] < (i+1)*NN_INPUT_DATA_LENGTH):
    #         annotation_pointers_pos.append(annotation.sample[ann_ptr] - i*NN_INPUT_DATA_LENGTH)
    #         ann_ptr += 1
    #         # Check for second peak
    #         if(i*NN_INPUT_DATA_LENGTH <= annotation.sample[ann_ptr] < (i+1)*NN_INPUT_DATA_LENGTH):
    #             annotation_pointers_neg.append(annotation.sample[ann_ptr] - i*NN_INPUT_DATA_LENGTH)
    #             ann_ptr += 1
    #         else:
    #             annotation_pointers_neg.append(0) 

    #     else:
    #         annotation_pointers_pos.append(0) 
    #         annotation_pointers_neg.append(0) 

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
training_input_dat = input_dat_1
training_output_dat = output_dat_1

for x in range(len(INPUT_FILE_PATH)-1):
    data_file, annotation_file = Read_Data_from_wfd(INPUT_FILE_PATH[x+1])
    input_dat_1, output_dat_1 = pre_process_data(data_file, annotation_file)
    training_input_dat = np.concatenate((training_input_dat, input_dat_1), axis=0)
    training_output_dat = np.concatenate((training_output_dat, output_dat_1), axis=0)

# Create model
model = tf.keras.Sequential()

##########
# MLP
##########
model.add(Dense(128, input_dim=NN_INPUT_DATA_LENGTH, activation='linear'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='softmax'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='relu'))
model.compile(loss='mean_squared_error',
              optimizer='RMSprop',
              metrics=['accuracy'])


history = model.fit(training_input_dat, training_output_dat,
          epochs=5000, validation_split=0.1,
          batch_size=2048)

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model)
  
model.summary()
model.save('current_run.h5')          


#score = model.evaluate(test_input_dat, test_output_dat)

predicted_out = model.predict(training_input_dat)

print(predicted_out)
print(training_output_dat)

plot_matplotlib_model(history)

# for i in range(0, predicted_out.size):
#     if(predicted_out[i] > 0.5):
#         print("yay")




