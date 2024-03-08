from sklearn.datasets import make_blobs #used to created clustarable datasets
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
# contains 20x20 images 
X = np.load("data/X.npy")
y = np.load("data/y.npy")
model = Sequential(
    [               
        ### START CODE HERE ### 
        tf.keras.Input(shape=(400,)),
        Dense(units=25,activation="relu"),
        Dense(units=15,activation="relu"),
        Dense(units=10, activation="linear"),
        ### END CODE HERE ### 
    ], name = "my_model" 
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.Adam(0.001)
)
model.fit(X,y,epochs=40)
res = model.predict(X)
n= len(X)
for i in range(5):
    print(np.argmax(res[i*899]), y[i*899])