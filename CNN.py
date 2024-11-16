"""
Notebook GPU issues need to run in Colab
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images = train_images/255.0
test_images = test_images/255.0


model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu',input_shape=(32,32,3)))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs=20)

loss, accuracy = model.evaluate(test_images, test_labels)
print('loss = ', loss)
print('accuracy = ', accuracy)

test_batch = test_images[:10]
preds = model.predict(test_batch)

print('preds = ', preds)
print()

for i in range(0,10):
  print(np.argmax(preds[i]))

