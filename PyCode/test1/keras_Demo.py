from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


print(tf.version.VERSION)
print(tf.keras.__version__)


# Generate dummy data
x_train = np.random.random((1000, 1))
y_train = np.random.randint(2, size=(1000, 1))

print(x_train.shape)
print(y_train.shape)

x_test = np.random.random((100, 1))
y_test = np.random.randint(2, size=(100, 1))

model = tf.keras.Sequential()

model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)

# Data for plotting
#s = 1 + np.sin(2 * np.pi * t)

# fig, ax = plt.subplots()
# ax.plot(x_train)

# ax.set(xlabel='time (s)', ylabel='voltage (mV)',
#        title='About as simple as it gets, folks')
# ax.grid()

# plt.show()

