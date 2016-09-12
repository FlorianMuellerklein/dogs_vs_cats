import time
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, MinMaxScaler

from keras.regularizers import l2
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Activation, Dropout, Flatten, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, Adam

import multiprocessing as mp

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
crop_features = cropPIXELS * cropPIXELS * 3
scaler = StandardScaler()
minmax = MinMaxScaler()
label_enc = LabelEncoder()

# set up training params
ITERS = 150
BATCHSIZE = 64
LR_SCHEDULE = {
    0: 0.003,
    75: 0.0003,
    125: 0.0001
}

def fast_warp(img, tf, output_shape, mode='reflect'):
    return transform._warps_cy._warp_fast(img, tf.params, output_shape=output_shape, mode=mode)


class threaded_batch_iter(object):
    '''
    Batch iterator to make transformations on the data
    '''
    def __init__(self, batchsize):
        self.batchsize = batchsize

    def __call__(self, X, y):
        self.X, self.y = X, y
        return self

    def __iter__(self):
        '''
        multi thread the iter so that the GPU does not have to wait for the CPU to process data
        '''
        q = mp.Queue(maxsize=128)

        def _gen_batches():
            num_samples = len(self.X)
            idx = np.random.permutation(num_samples)
            batches = range(0, num_samples - self.batchsize + 1, self.batchsize)
            for batch in batches:
                X_batch = self.X[idx[batch:batch + self.batchsize]]
                y_batch = self.y[idx[batch:batch + self.batchsize]]

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

                yield [X_batch_aug, y_batch]

        def _producer(_gen_batches):
            batch_gen = _gen_batches()
            for data in batch_gen:
                q.put(data, block=True)
            q.put(None)
            q.close()

        thread = mp.Process(target=_producer, args=[_gen_batches])
        thread.daemon = True
        thread.start()

        for data in iter(q.get, None):
            yield data[0], data[1]

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


def load_data_cv(train_img_path, train_targs_path):

    print('read data')
    # reading training data
    training_inputs = np.load(train_img_path)
    training_targets = np.load(train_targs_path)

    # split training labels and pre-process them
    #training_targets = label_enc.fit_transform(training_targets)
    #training_targets = training_targets.astype('int32')
    #training_targets = np_utils.to_categorical(training_targets)
    print 'training targets shape'
    print training_targets.shape

    # split training inputs and scale data 0 to 1
    #training_inputs = training[:,0:num_features].astype('float32')
    training_inputs = training_inputs / 255.

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(training_inputs, training_targets, test_size = 0.05)

    print 'train size:', X_train.shape[0], 'eval size:', X_test.shape[0]

    # reshaping training and testing data so it can be feed to convolutional layers
    #x_train = x_train.reshape(x_train.shape[0], 3, PIXELS, PIXELS)
    #x_test = x_test.reshape(x_test.shape[0], 3, PIXELS, PIXELS)

    # check to see whether everything loaded correctly
    print np.amax(X_train[0])
    #for i in range(0,5):
    #    j = randint(0, x_train.shape[0])
    #    plot_sample(x_train[j])
    #    print y_train[j]

    X_test = resize_valid_set(X_test)

    return X_train, X_test, y_train, y_test

def load_data_train(train_path):

    print('read data')
    # reading training data
    training = np.load(train_path)

    # split training labels and pre-process them
    training_targets = training[:,num_features]
    training_targets = label_enc.fit_transform(training_targets)
    training_targets = training_targets.astype('int32')
    #training_targets = np_utils.to_categorical(training_targets)

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
    testing_inputs = testing[:,0:crop_features].astype('float32')
    testing_inputs = testing_inputs / 255.

    # reshaping training and testing data so it can be feed to convolutional layers
    testing_inputs = testing_inputs.reshape(testing_inputs.shape[0], 3, cropPIXELS, cropPIXELS)

    return testing_inputs

def build_model():
    '''
    VGG style CNN. Using either ReLU, PReLU or LeakyReLU in the fully connected layers
    '''
    print('creating the model')
    cnn_input = Input(shape=(3,cropPIXELS,cropPIXELS), name='Input', dtype='float32')

    conv1a = Convolution2D(32, 3,3, init='he_normal', activation='relu')(cnn_input)
    conv1b = Convolution2D(32, 3,3, init='he_normal', activation='relu')(conv1a)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1b)

    conv2a = Convolution2D(64, 3,3, init='he_normal', activation='relu')(pool1)
    conv2b = Convolution2D(64, 3,3, init='he_normal', activation='relu')(conv2a)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2b)

    conv3a = Convolution2D(128, 3,3, init='he_normal', activation='relu')(pool2)
    conv3b = Convolution2D(128, 3,3, init='he_normal', activation='relu')(conv3a)
    conv3c = Convolution2D(128, 3,3, init='he_normal', activation='relu')(conv3b)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3c)

    conv4a = Convolution2D(256, 3,3, init='he_normal', activation='relu')(pool3)
    conv4b = Convolution2D(256, 3,3, init='he_normal', activation='relu')(conv4a)
    conv4c = Convolution2D(256, 3,3, init='he_normal', activation='relu')(conv4b)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4c)

    # convert convolutional filters to flat so they can be feed to fully connected layers
    flatten = Flatten()(pool3)

    # fully connected
    fc1 = Dense(1024, init= 'he_normal', activation='relu')(flatten)
    dropout1 = Dropout(p=0.5)(fc1)

    fc2 = Dense(1024, init= 'he_normal', activation='relu')(dropout1)
    dropout2 = Dropout(p=0.5)(fc2)

    pred = Dense(1, activation='sigmoid')(dropout2)
    model = Model(input=cnn_input, output=pred)

    # optimizers
    sgd = SGD(lr=0.003, decay=0.0001, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

def pre_resnet(n=2):
    '''
    Adapted from https://github.com/Lasagne/Recipes/tree/master/papers/deep_residual_learning.
    Tweaked to be consistent with 'Identity Mappings in Deep Residual Networks', Kaiming He et al. 2016 (https://arxiv.org/abs/1603.05027)
    Forumala to figure out depth: 6n + 2
    '''
    n_filters = {0:64, 1:64, 2:128, 3:256}
    def residual_block(l, increase_dim=False, first=False, filters=16):
        if increase_dim:
            first_stride = (2,2)
        else:
            first_stride = (1,1)

        if first:
            pre_act = l
        else:
            # BN -> ReLU
            bn = BatchNormalization(axis=1)(l)
            pre_act = Activation('relu')(bn)

        conv_1 = Convolution2D(filters, 3,3, init='he_normal', border_mode='same', subsample=first_stride, activation='linear')(pre_act)
        bn_1 = BatchNormalization(axis=1)(conv_1)
        relu_1 = Activation('relu')(bn_1)
        conv_2 = Convolution2D(filters, 3,3, init='he_normal', border_mode='same', activation='linear')(relu_1)

        # add shorcut
        if increase_dim:
            # projection shortcut
            projection = Convolution2D(filters, 1,1, subsample=(2,2), border_mode='same', activation='linear')(pre_act)
            block = merge([conv_2, projection], mode='sum')
        else:
            block = merge([conv_2, pre_act], mode='sum')

        return block

    cnn_input = Input(shape=(3,cropPIXELS,cropPIXELS), name='Input', dtype='float32')

    l = Convolution2D(n_filters[0], 5,5, init='he_normal', border_mode='same', activation='linear')(cnn_input)
    l = BatchNormalization(axis=1)(l)
    l = Activation('relu')(l)
    l = MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='same')(l)

    # first stack of residual blocks, output is 32 x 64 x 64
    l = residual_block(l, first=True, filters=n_filters[1])
    for _ in range(1,n):
        l = residual_block(l, filters=n_filters[1])

    # second stack of residual blocks, output is 64 x 32 x 32
    l = residual_block(l, increase_dim=True, filters=n_filters[2])
    for _ in range(1,(n+2)):
        l = residual_block(l, filters=n_filters[2])

    # third stack of residual blocks, output is 128 x 16 x 16
    l = residual_block(l, increase_dim=True, filters=n_filters[3])
    for _ in range(1,(n+1)):
        l = residual_block(l, filters=n_filters[3])

    l = BatchNormalization(axis=1)(l)
    l = Activation('relu')(l)

    avg_pool = GlobalAveragePooling2D()(l)

    pred = Dense(1, activation='sigmoid')(avg_pool)
    model = Model(input=cnn_input, output=pred)

    # optimizers
    sgd = SGD(lr=0.01, decay=0.0001, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

def main():

    # switch the commented lines here to alternate between CV testing and making kaggle submission
    X_train, X_test, y_train, y_test = load_data_cv('data/train_color.npy', 'data/train_color_labels.npy')
    print 'valid shape:', X_test.shape, 'train shape:', X_train.shape
    #x_train, y_train = load_data_train('data/train_color.npy')

    model = pre_resnet(n=2)

    # load batch_iter
    batch_iter = threaded_batch_iter(batchsize=BATCHSIZE)

    print("Starting training")
    # batch iterator with 300 epochs
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    try:
        for epoch in range(ITERS):
            # change learning rate according to schedule
            if epoch in LR_SCHEDULE:
                model.optimizer.lr.set_value(LR_SCHEDULE[epoch])
            start = time.time()
            #loss = batch_iterator(x_train, y_train, 64, model)
            batch_loss = []
            batch_acc = []
            for X_batch, y_batch in batch_iter(X_train, y_train):
                loss, acc_t = model.train_on_batch(X_batch, y_batch)
                train_loss.append(loss)
                train_acc.append(acc_t)
                batch_loss.append(loss)
                batch_acc.append(acc_t)

            #train_loss.append(loss)
            v_loss, v_acc = model.evaluate(X_test, y_test, batch_size=BATCHSIZE, verbose = 0)
            valid_loss.append(v_loss)
            valid_acc.append(v_acc)
            end = time.time() - start
            print epoch, '| Tloss:', np.round(np.mean(batch_loss), decimals = 3), '| Tacc:', np.round(np.mean(batch_acc), decimals = 3), '| Vloss:', np.round(v_loss, decimals = 3), '| Vacc:', np.round(v_acc, decimals = 3), '| time:', np.round(end, decimals = 1)
    except KeyboardInterrupt:
        pass

    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    valid_acc = np.array(valid_acc)
    sns.set_style("whitegrid")
    pyplot.plot(train_loss, linewidth = 3, label = 'train loss')
    pyplot.legend(loc = 2)
    pyplot.ylim([0,0.85])
    pyplot.show()

    pyplot.plot(valid_loss, linewidth = 3, label = 'valid loss')
    pyplot.legend(loc = 2)
    pyplot.ylim([0,0.85])
    pyplot.twinx()
    pyplot.plot(valid_acc, linewidth = 3, label = 'valid accuracy', color = 'r')
    pyplot.grid()
    pyplot.ylim([0,1])
    pyplot.legend(loc = 1)
    pyplot.show()

    print("Saving model ... ")
    model.save_weights('dogs_v_cats_cnn')

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
