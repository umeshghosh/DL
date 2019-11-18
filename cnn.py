import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()


# scales data between 0 and 1
x_train = X_train.reshape(60000,28,28,1)/255.
x_test  =  X_test.reshape(10000,28,28,1)/255. 

m = Sequential()
m.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
m.add(Conv2D(16, (3, 3), activation='relu'))
m.add(MaxPooling2D(pool_size=(2, 2)))
m.add(Dropout(0.25))
m.add(Flatten())
m.add(Dense(10, activation='relu'))
m.add(Dropout(0.5))
m.add(Dense(10, activation='softmax'))

m.compile(loss='sparse_categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

m.summary()

m.fit(x_train, y_train, epochs=1)  
