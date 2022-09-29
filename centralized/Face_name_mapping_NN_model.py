# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 23:27:18 2022

@author: rajan
"""

import pickle
import tensorflow as tf

imgs = []

# Load the pickle fiole with 128 dim vectors 

with open("test", "rb") as fp:  
  imgs = pickle.load(fp)
  
# Manually generating encodings for images (instead of reading filename) since each image has a unique name
y = list(range(1, len(imgs)+1))


len(imgs)

# Just double checking if any vector isnt of length 128

for i in range(len(imgs)):
  if len(imgs[i])!=128:
    print(i)
print("All okay!")


# Onehot encode y

#Import necessary libraries

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# one hot encode
y_enc = to_categorical(y)
y_enc.shape


# Scale image vector values 

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(imgs)



# Build NN to predcit name 

ann1 = tf.keras.models.Sequential()

# ann1.add(tf.keras.layers.Dense(units=32, activation='relu'))


# ann1.add(tf.keras.layers.Dense(units=32, activation='relu'))
# ann1.add(tf.keras.layers.Dense(units=32, activation='relu'))
# ann1.add(tf.keras.layers.Dense(units=32, activation='relu'))

#ann.add(tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(-0.1)))


#output layer
ann1.add(tf.keras.layers.Dense(units=993, activation='softmax'))


opt = tf.keras.optimizers.Adam(learning_rate=0.001)
ann1.compile(optimizer=opt, loss= tf.keras.losses.categorical_crossentropy, metrics = ['accuracy'])



# Train model

hist_ensemble_1 = ann1.fit(X_train, y_enc, batch_size= len(X_train), epochs = 100)



# Evaluate model

ann1.evaluate(X_train,y_enc)



