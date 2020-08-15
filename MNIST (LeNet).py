""" MNIST data base consists of 60,000 training samples and 10,000 testing samples 
Image size is 28x28, greyscale """

# LeNet 

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Input
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

""" x_train and x_test are the input images for training and test data respectively
y_train and y_test are the output/ground truth for training and test data respectively
x_train dim: 60,000:28:28 """

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255 # Normalizing, white pixel value = 255
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# reshaping x_train and x_test to 4 dimensional array: 60,000:28:28:1

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# to_categorical converts labels into binary labels using one hot encoding

""" Adding all the layers"""


inputs = Input(shape=x_train.shape[1:])

x = Conv2D(filters=6, kernel_size=(5,5), strides=1, padding='valid', activation='relu', input_shape=x_train.shape[1:])(inputs)
x = MaxPool2D(pool_size=(2,2), strides=2)(x)
x = Conv2D(filters=16, kernel_size=(5,5), strides=1, padding='valid', activation='relu')(x)
x = MaxPool2D(pool_size=(2,2), strides=2)(x)
x = Flatten()(x)
x = Dense(120, activation='relu')(x)
x = Dense(84, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 3
batch_size = 32
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=None, validation_split=0, validation_data=(x_test, y_test))


"""Visualizing"""

#Training/Testing accuracy over epochs
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='testing accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

#Loss Curve
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='testing loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
