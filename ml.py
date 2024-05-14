import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.activations import linear
import keras
import numpy as np


tf.random.set_seed(1)

#X = np.array([[1], [3], [2], [10], [4], [7], [8]]) 
#y = np.array([[3, 9, 6, 30, 12, 21, 24]]).T


def my_generator(amount,power):
    for i in range(amount):
        if power != 0:
            yield i * power
        else:
            yield i   


X = np.fromiter(my_generator(7,0), dtype=int)#newshape=(7)
y = np.fromiter(my_generator(7,3), dtype=int)#,newshape=(7)
print(X,y)
#print(x,y)

model = Sequential([

    keras.Input(shape=(1,),dtype=int),
    Dense(1, activation='linear')

])

w1, w0 = model.get_weights()
model.compile(optimizer='sgd', loss='mse', metrics=['mae'])
model.fit(X,y, epochs=100)
print(X[:1])
l = np.array(X)
print(model.predict(l[:1]))
#print(model.predict(np.array([[10],[30]])))

model.summary()