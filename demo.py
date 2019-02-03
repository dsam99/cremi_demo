import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout, MaxPooling2D

# creating our keras model
model = Sequential()

# simple feed-forward structure
model.add(Dense(250, activation='softmax',
				input_shape=(1250, 1250)))
model.add(Dropout(0.5))
model.add(Dense(1250, activation='softmax'))

# Compiling model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# reading h5 file
f_train = h5py.File("CREMI/sample_a_train.hdf")
f_test = h5py.File("CREMI/sample_a_test.hdf")

# converting training data into workable data format
pixels = f_train['volumes']['raw']
labels = f_train['volumes']['labels']['neuron_ids']

# converting traning data into numpy arrays
train_data = np.array(pixels[::].tolist())
train_labels = np.array(labels[::].tolist())

# converting testing pixels into workable data format
pixels = f_test['volumes']['raw']
test_data = np.array(pixels[::].tolist())

'''
Data format in row-major form
(depth, height , width) in nm
'''

model.fit(train_data,
          train_labels,
          batch_size=10,
          shuffle=True,
          epochs=3)
