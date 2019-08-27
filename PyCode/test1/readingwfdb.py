import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

import wfdb
import wfdb.io as wfdbio

INPUT_FILE_PATH = '..\\..\\mit-bih-database\\100'

record = wfdbio.rdsamp(INPUT_FILE_PATH, channels = [0])
annotation = wfdb.rdann(INPUT_FILE_PATH, 'pwave')
#print(record)
#print(record.p_signal)
#print(annotation.sample)

annotation_timed = []
ann_ptr = 0

SIZE_PLOT = record[0].size

for i in range(0, record[0].size):
    if(ann_ptr < annotation.sample.size):
        if(i == annotation.sample[ann_ptr]):
            annotation_timed.append(1)
            ann_ptr += 1
        else:
            annotation_timed.append(0) 
    else:
        annotation_timed.append(0)


print(annotation_timed)
input_dat = record[0][:SIZE_PLOT].squeeze().tolist()

t = range(0,SIZE_PLOT)

plt.plot(t, annotation_timed, t, input_dat)

plt.show()

# wfdb.plot_wfdb(record=record, annotation=annotation,
#                title='Record 100 from MIT-BIH Arrhythmia Database',
#                time_units='seconds')


