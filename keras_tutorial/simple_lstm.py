import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
import keras

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape[1:])
print(x_train[0].shape)
x_train=x_train/255.0
x_test=x_test/255.0
model=Sequential()

model.add(LSTM(128,input_shape=(x_train.shape[1:]),activation="relu",return_sequences="True"))
model.add(Dropout(0.2))

model.add(LSTM(128,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(10,activation="softmax"))


# opt=tf.keras.optimizers.Adam(lr=1e-3,decay=1e-5)
# model.compile(loss="sparse_categorical_crossentropy",optimizer=opt,matrics=["accuracy"])

# Set Optimizer
opt = keras.optimizers.adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)
model.fit(x_train,y_train,epochs=3,validation_data=(x_test,y_test))
