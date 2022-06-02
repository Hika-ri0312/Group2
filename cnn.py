import numpy as np
import dataset
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from tensorflow import keras

(x_train, x_test), (y_train, y_test) = dataset.get_dataset()
class_names = [Hirosada, Kogyo, Kunichika, Kunisada_1st, Kunisada_2ed, Kunisada_3rd, Kuniyoshi, Toyokuni_1st,Toyokuni_3rd, Yoshitaki]

print("Train samples:", x_train.shape, x_test.shape)
print("Test samples:", y_train, y_test.shape)

x_train = x_train.reshape((220, 32, 32, 3))
y_train = y_train.reshape((70, 32, 32, 3))

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

'''x_train = x_train/255.0
y_train = y_train/255.0
