import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequantial
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmod
# tf.autograph.set_verbosity(0 )
layer_1 = Dense(units=3,activation="sigmoid")
layer_2 = Dense(units=1, activation="sigmoid")
model = Sequantial([layer_1, layer_2])

x = np.array([1,0,0,1])
model.compile()
model.fit(x,y)
model.pre