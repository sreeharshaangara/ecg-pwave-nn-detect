from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import threading
import serial
import struct

import collections
import random
import time
import math
import numpy as np

class DynamicPlotter():
    input_buffer = np.zeros(320)
    output_buffer = np.zeros(320)
    plot_pointer = 0
    data_rdy = 0
    timehist = time.time()
    timechk = 0.80

    def __init__(self, sampleinterval=0.1, timewindow=20., size=(600,350)):
        # Data stuff
        pg.setConfigOptions(antialias=True)

        self._interval = int(sampleinterval*1000)
        self._bufsize = int(timewindow/sampleinterval)
        self.databuffer_1 = collections.deque([0.0]*self._bufsize, self._bufsize)
        self.databuffer_2 = collections.deque([0.0]*self._bufsize, self._bufsize)
        self.x = np.linspace(-timewindow, 0.0, self._bufsize)
        self.y_1 = np.zeros(self._bufsize, dtype=np.float)
        self.y_2 = np.zeros(self._bufsize, dtype=np.float)
        
        # PyQtGraph stuff
        self.app = QtGui.QApplication([])
        self.plt = pg.plot(title='Dynamic Plotting with PyQtGraph')
        self.plt.resize(*size)
        self.plt.showGrid(x=True, y=True)
        self.plt.setLabel('left', 'amplitude', 'V')
        self.plt.setLabel('bottom', 'time', 's')
        self.curve1 = self.plt.plot(self.x, self.y_1, pen=(255,0,0))
        self.curve2 = self.plt.plot(self.x, self.y_2, pen=(0,255,0))

        
        # QTimer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateplot)
        # self.timer.start(31)
        self.timer.start(30)

    def update_array(self, in_arr, out_arr):
        self.input_buffer = in_arr
        self.output_buffer = out_arr
        self.data_rdy = 1
        self.timechk = (time.time() - self.timehist)
        self.timehist = time.time()

    def getdata(self):
        new = np.zeros(2)
        new[0] = self.input_buffer[self.plot_pointer] 
        new[1] = self.output_buffer[self.plot_pointer]
        if(self.plot_pointer == 319):        
            self.plot_pointer = 0
            self.data_rdy = 0
        else:
            self.plot_pointer += 1
        return new

    def updateplot(self):

        for x in range(13):
            if(self.data_rdy == 1):
                newval = self.getdata()
                self.databuffer_1.append(newval[0])
                self.y_1[:] = self.databuffer_1
                self.databuffer_2.append(newval[1])
                self.y_2[:] = self.databuffer_2

        self.curve1.setData(self.x, self.y_1)
        self.curve2.setData(self.x, self.y_2)
        self.app.processEvents()

    def run(self):
        self.app.exec_()

def handle_data(data, qtplot):
    # Check for valid data packet header
    if((data[0] == 23) and (data[1] == 171)):

        #Peak location
        peakloc_1_raw = struct.unpack('f',bytearray(data[2:6]))[0]
        peakloc_2_raw = struct.unpack('f',bytearray(data[6:10]))[0]
        print(peakloc_1_raw)
        print(peakloc_2_raw)
        # peakloc_2_raw = struct.unpack('f',bytearray(data[10:14]))[0]
        
        input_arr = []

        for x in range(320):
            
            start = 10 + 4*x
            end = 10 + 4*x + 4 
            
            input_arr.append(struct.unpack('i',bytearray(data[start:end]))[0])

        # peakloc_1_raw = 0.1
        # peakloc_2_raw = 0.9
        
        #print(input_arr)
        #Form output array
        peak_arr_out = np.zeros(320)
        if(peakloc_1_raw != 0):
            if(peakloc_1_raw > 1):
                peakloc_1_raw = 0.99
            peak_arr_out[int(peakloc_1_raw*320)+1] = input_arr[int(peakloc_1_raw*320)+1] #2000
        if(peakloc_2_raw != 0):    
            if(peakloc_2_raw > 1):
                peakloc_2_raw = 0.99

            peak_arr_out[int(peakloc_2_raw*320)+1] = input_arr[int(peakloc_2_raw*320)+1] # 2000
        
        #print(peak_arr_out)

        qtplot.update_array(input_arr, peak_arr_out)

    else:
        # Not a data packet
        print(data)

def read_from_port(ser, qtplot):
    while(1):
        if (ser.inWaiting()):
            # Wait till 1000 bytes or timeout
            reading = ser.read(1300)
            #print(reading)
            handle_data(reading, qtplot)

if __name__ == '__main__':


    m = DynamicPlotter(sampleinterval=(1/320), timewindow=10.)
    serial_port = serial.Serial("COM6", 115200, timeout=0.15)
    serial_port.reset_input_buffer()
    thread = threading.Thread(target=read_from_port, args=(serial_port,m, ))
    thread.start()
    
    m.run()