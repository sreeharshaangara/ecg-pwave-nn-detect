import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sys
import wfdb
import wfdb.io as wfdbio


INPUT_FILE_PATH = ['..\\..\\mit-bih-database\\100', '..\\..\\mit-bih-database\\101']
NN_INPUT_DATA_LENGTH = 320
TRAINING_DATA_LIMIT = 640

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



#########
# Main
#########
data_file, annotation_file = Read_Data_from_wfd(INPUT_FILE_PATH[0])
input_dat_1, output_dat_1 = pre_process_data(data_file, annotation_file)

print(input_dat_1[0])

np.savetxt("foo.txt", input_dat_1,fmt="%0.7f", delimiter=",")
