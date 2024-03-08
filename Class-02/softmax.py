from sklearn.datasets import make_blobs #used to created clustarable datasets
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# make  dataset for example
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)

model = Sequential([
    Dense(units=25,activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=4, activation='softmax') # output layer with 4 neurons (for  4 clusters)
])
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer = tf.keras.optimizers.Adam(0.001))
model.fit(X_train,y_train,epochs=10)

print(model.predict(X_train[:2])) #the values near to 1 is class it belongs to, there are 4 clases here
