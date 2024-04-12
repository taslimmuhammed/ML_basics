import numpy as np 
import numpy.ma as ma
import pandas as pd
import tensorflow  as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models  import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from recsysNN_utils import *
import tabulate

num_outputs = 32
num_user_features = 256
num_item_features = 128
tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=num_outputs)
])

item_NN = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=num_outputs)
])
# create the user input and point to the base network
input_user = tf.keras.layers.input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

input_item = tf.keras.layers.input(shape=(num_item_features)) #
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)
output = tf.keras.layers.Dot(axes=1)([vu,vm])
model = tf.keras.Model([input_user,input_item],output)
model.summary()
model.compile(optimizer = tf.keras.optimizers.Adam(0.01),loss=tf.keras.losses.MeanSquaredError())
# model.evaluate(X_train,y_train)