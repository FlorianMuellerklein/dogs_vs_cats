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

PIXELS = 192
cropPIXELS = 168
imageSize = PIXELS * PIXELS
num_features = imageSize * 3
scaler = StandardScaler()
minmax = MinMaxScaler()
label_enc = LabelEncoder()

def fast_warp(img, tf, output_shape, mode='nearest'):
    return transform._warps_cy._warp_fast(img, tf.params, output_shape=output_shape, mode=mode)

def batch_iterator(data, y, batchsize, model):
    '''
    Data augmentation batch iterator for feeding images into CNN.
    This example will randomly rotate all images in a given batch between -30 and 30 degrees
    and to random translations between -24 and 24 pixels in all directions.
    Random zooms between 1 and 1.3.
    Random shearing between -10 and 10 degrees.
    '''

    n_samples = data.shape[0]
    loss = []
    for i in range((n_samples + batchsize -1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch = data[sl]
        y_batch = y[sl]

        # set empty copy to hold augmented images so that we don't overwrite
        X_batch_aug = np.empty(shape = (X_batch.shape[0], 3, cropPIXELS, cropPIXELS), dtype = 'float32')

        # random rotations betweein -8 and 8 degrees
        dorotate = randint(-30,30)

        # random translations
        trans_1 = randint(-12,12)
        trans_2 = randint(-12,12)
        crop_amt = ((12 - trans_1, 12 + trans_1), (12 - trans_2, 12 + trans_2), (0,0))

        # random zooms
        zoom = uniform(1, 1.3)

        # shearing
        shear_deg = uniform(-10, 10)

        # set the transform parameters for skimage.transform.warp
        # have to shift to center and then shift back after transformation otherwise
        # rotations will make image go out of frame
        center_shift   = np.array((PIXELS, PIXELS)) / 2. - 0.5
        tform_center   = transform.SimilarityTransform(translation=-center_shift)
        tform_uncenter = transform.SimilarityTransform(translation=center_shift)

        tform_aug = transform.AffineTransform(rotation = np.deg2rad(dorotate),
                                              #translation = (trans_1, trans_2),
                                              shear = np.deg2rad(shear_deg),
                                              scale = (1/zoom, 1/zoom))

        tform = tform_center + tform_aug + tform_uncenter

        r_intensity = randint(0,1)
        g_intensity = randint(0,1)
        b_intensity = randint(0,1)
        intensity_scaler = uniform(-0.25, 0.25)

        # images in the batch do the augmentation
        for j in range(X_batch.shape[0]):
            img = X_batch[j]
            img = img.transpose(1, 2, 0)
            img_aug = np.zeros((PIXELS, PIXELS, 3))
            for k in range(0,3):
                img_aug[:, :, k] = fast_warp(img[:, :, k], tform, output_shape = (PIXELS, PIXELS))

            img_aug = crop(img_aug, crop_amt)

            if r_intensity == 1:
                img_aug[:, :, 0] = img_aug[:, :, 0] + (np.std(img_aug[:, :, 0]) * intensity_scaler)
            if g_intensity == 1:
                img_aug[:, :, 1] = img_aug[:, :, 1] + (np.std(img_aug[:, :, 1]) * intensity_scaler)
            if b_intensity == 1:
                img_aug[:, :, 2] = img_aug[:, :, 2] + (np.std(img_aug[:, :, 2]) * intensity_scaler)

            X_batch_aug[j] = img_aug.transpose(2, 0, 1)

        # Flip half of the images in this batch at random:
        bs = X_batch.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_batch_aug[indices] = X_batch_aug[indices, :, :, ::-1]

        #plot_sample(X_batch_aug[0])

        # fit model on each batch
        loss.append(model.train(X_batch_aug, y_batch))

    return np.mean(loss)

def plot_sample(x):
    img = x.transpose(1, 2, 0)
    imshow(img)
    pyplot.show()

def resize_valid_set(valid_data):
    n_samples = valid_data.shape[0]
    valid_resized = np.empty(shape = (valid_data.shape[0], 3, cropPIXELS, cropPIXELS), dtype = 'float32')
    for i in range(n_samples):
        img = valid_data[i]
        img = img.transpose(1, 2, 0)
        img = transform.resize(img, output_shape = (cropPIXELS, cropPIXELS, 3))
        img = img.transpose(2, 0, 1)
        valid_resized[i] = img
    return valid_resized


def load_data_cv(train_path):

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

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(training_inputs, training_targets, test_size = 0.05)

    print 'train size:', x_train.shape[0], 'eval size:', x_test.shape[0]

    # reshaping training and testing data so it can be feed to convolutional layers
    x_train = x_train.reshape(x_train.shape[0], 3, PIXELS, PIXELS)
    x_test = x_test.reshape(x_test.shape[0], 3, PIXELS, PIXELS)

    # check to see whether everything loaded correctly
    print np.amax(x_train[0])
    #for i in range(0,5):
    #    j = randint(0, x_train.shape[0])
    #    plot_sample(x_train[j])
    #    print y_train[j]

    x_test = resize_valid_set(x_test)

    return x_train, x_test, y_train, y_test

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

def build_model():
    '''
    VGG style CNN. Using either ReLU, PReLU or LeakyReLU in the fully connected layers
    '''
    print('creating the model')
    model = Sequential()

    model.add(Convolution2D(32,3, 3,3, init='glorot_uniform', activation = 'relu', border_mode='full'))
    model.add(Convolution2D(32,32, 3,3, init='glorot_uniform', activation = 'relu'))
    model.add(MaxPooling2D(poolsize=(2,2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(64,32, 3,3, init='glorot_uniform', activation = 'relu', border_mode='full'))
    model.add(Convolution2D(64,64, 3,3, init='glorot_uniform', activation = 'relu'))
    model.add(MaxPooling2D(poolsize=(2,2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(128,64, 3,3, init='glorot_uniform', activation = 'relu', border_mode='full'))
    model.add(Convolution2D(128,128, 3,3, init='glorot_uniform', activation = 'relu'))
    model.add(MaxPooling2D(poolsize=(2,2)))
    model.add(Dropout(0.1))

    # convert convolutional filters to flat so they can be feed to fully connected layers
    model.add(Flatten())

    model.add(Dense(56448,2048, init='glorot_uniform', activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2048,2048, init='glorot_uniform', activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2048,2, init='glorot_uniform'))
    model.add(Activation('softmax'))

    # setting sgd optimizer parameters
    sgd = SGD(lr=0.03, decay=0.001, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def main():

    # switch the commented lines here to alternate between CV testing and making kaggle submission
    x_train, x_test, y_train, y_test = load_data_cv('data/train_color.npy')
    print 'valid shape:', x_test.shape, 'train shape:', x_train.shape
    #x_train, y_train = load_data_train('data/train_color.npy')

    model = build_model()

    print("Starting training")
    # batch iterator with 300 epochs
    train_loss = []
    valid_loss = []
    valid_acc = []
    for i in range(150):
        start = time.time()
        loss = batch_iterator(x_train, y_train, 32, model)
        train_loss.append(loss)
        valid_avg = model.evaluate(x_test, y_test, show_accuracy = True, verbose = 0)
        valid_loss.append(valid_avg[0])
        valid_acc.append(valid_avg[1])
        end = time.time() - start
        print 'iter:', i, '| Tloss:', np.round(loss, decimals = 3), '| Vloss:', np.round(valid_avg[0], decimals = 3), '| Vacc:', np.round(valid_avg[1], decimals = 3), '| time:', np.round(end, decimals = 1)

    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    valid_acc = np.array(valid_acc)
    sns.set_style("whitegrid")
    pyplot.plot(train_loss, linewidth = 3, label = 'train loss')
    pyplot.plot(valid_loss, linewidth = 3, label = 'valid loss')
    pyplot.legend(loc = 2)
    pyplot.ylim([0,0.85])
    pyplot.twinx()
    pyplot.plot(valid_acc, linewidth = 3, label = 'valid accuracy', color = 'r')
    pyplot.grid()
    pyplot.ylim([0,1])
    pyplot.legend(loc = 1)
    pyplot.show()


    #print("Generating predections")
    #x_train = None
    #x_test = load_data_test('data/test_color.npy')
    #preds = model.predict_classes(x_test, verbose=0)
    #preds = label_enc.inverse_transform(preds).astype(int)

    #submission = pd.read_csv('data/sampleSubmission.csv', dtype = int)
    #submission['label'] = preds
    #submission.to_csv('preds/DogsvsCats_cnn.csv', index = False)

if __name__ == '__main__':
    main()
