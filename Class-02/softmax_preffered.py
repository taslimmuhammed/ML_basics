from sklearn.datasets import make_blobs #used to created clustarable datasets
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)

model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=25, activation='relu'),
    Dense(units=4, activation='linear')
])
# Compile the model with a suitable optim
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # from_legits tells to include softmax in loss calculation
                optimizer = tf.keras.optimizers.Adam(0.001)) 
model.fit(X_train,y_train,epochs=10)
res = model.predict(X_train[:5])
  
for i in range(5):
    print( f"{res[i]}, category: {np.argmax(res[i])}") # max value among 4 of them, is the predicted class