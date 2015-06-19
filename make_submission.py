import time
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, MinMaxScaler

from keras.regularizers import l2
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD

from random import randint, uniform

import seaborn as sns
from matplotlib import pyplot
from skimage.io import imshow
from skimage.util import crop
from skimage import transform, filters, exposure

PIXELS = 168
imageSize = PIXELS * PIXELS
num_features = imageSize * 3
label_enc = LabelEncoder()

def batch_iterator(data, y, batchsize, model):
    
    n_samples = data.shape[0]
    loss = []
    for i in range((n_samples + batchsize -1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch = data[sl]
        y_batch = y[sl]

        model.train(X_batch, y_batch)


def load_data_test(test_path):

    print('read data')
    # read testing data
    testing = np.load(test_path).astype('float32')

    # split training inputs and scale data 0 to 1
    testing_inputs = testing[:,0:num_features].astype('float32')
    testing_inputs = testing_inputs / 255.

    # reshaping training and testing data so it can be feed to convolutional layers
    testing_inputs = testing_inputs.reshape(testing_inputs.shape[0], 3, PIXELS, PIXELS)

    return testing_inputs

def load_data_train(train_path):

    print('read data')
    # reading training data
    training = np.load(train_path)

    # split training labels and pre-process them
    training_targets = training[:,num_features]
    training_targets = label_enc.fit_transform(training_targets)
    training_targets = training_targets.astype('int32')
    training_targets = np_utils.to_categorical(training_targets)

    # split training inputs and scale data 0 to 1
    training_inputs = training[:,0:num_features].astype('float32')
    training_inputs = training_inputs / 255.

    # reshaping training and testing data so it can be feed to convolutional layers
    training_inputs = training_inputs.reshape(training_inputs.shape[0], 3, PIXELS, PIXELS)

    return training_inputs, training_targets

def build_model():
    '''
    VGG style CNN. Using either ReLU, PReLU or LeakyReLU in the fully connected layers
    '''
    print('creating the model')
    model = Sequential()

    model.add(Convolution2D(32,3, 3,3, init='glorot_uniform', activation='linear', border_mode='full'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Convolution2D(32,32, 3,3, init='glorot_uniform', activation='linear'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(poolsize=(2,2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(64,32, 3,3, init='glorot_uniform', activation='linear', border_mode='full'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Convolution2D(64,64, 3,3, init='glorot_uniform', activation='linear'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(poolsize=(2,2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128,64, 3,3, init='glorot_uniform', activation='linear', border_mode='full'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Convolution2D(128,128, 3,3, init='glorot_uniform', activation='linear'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Convolution2D(128,128, 3,3, init='glorot_uniform', activation='linear'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(poolsize=(2,2)))
    model.add(Dropout(0.3))

    # convert convolutional filters to flat so they can be feed to fully connected layers
    model.add(Flatten())

    model.add(Dense(51200,1024, init='glorot_uniform', activation='linear'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))

    model.add(Dense(1024,1024, init='glorot_uniform', activation='linear'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))

    model.add(Dense(1024,2, init='glorot_uniform'))
    model.add(Activation('softmax'))

    # setting sgd optimizer parameters
    sgd = SGD(lr=0.0001, decay=0.0001, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def main():

    model = build_model()

    print("loading model ... ")
    model.load_weights('dogs_v_cats_cnn')

    print('fine tuning model ... ')
    x_train, y_train = load_data_train('data/train_color_resize.npy')
    for i in range(5):
        batch_iterator(x_train, y_train, 32, model)

    print("Generating predections")
    x_train = None
    x_test = load_data_test('data/test_color.npy')
    preds = model.predict_classes(x_test, verbose=0)
    #preds = label_enc.inverse_transform(preds).astype(int)

    submission = pd.read_csv('data/sampleSubmission.csv', dtype = int)
    submission['label'] = preds
    submission.to_csv('preds/DogsvsCats_cnn.csv', index = False)

if __name__ == '__main__':
    main()
