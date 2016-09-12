import os
import numpy as np
import pandas as pd
import cPickle as pickle
from random import randint
from natsort import natsorted

from skimage import exposure
from matplotlib import pyplot
from skimage.io import imread
from skimage.io import imshow
from skimage.filters import sobel
from skimage import feature, transform

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

PATH = 'data/train'
PIXELS = 192

def plot_sample(x):
    img = x.reshape(maxPixel, maxPixel, 3)
    imshow(img)
    pyplot.show()

def load_images(path):
    print 'reading file names ... '
    names = [d for d in os.listdir (path) if d.endswith('.jpg')]
    names = natsorted(names)
    num_rows = len(names)
    print num_rows

    print 'making dataset ... '
    train_image = np.zeros((num_rows, 3, PIXELS, PIXELS), dtype=np.float32)
    levels = np.zeros((num_rows,), dtype=np.int32)
    file_names = []
    i = 0
    for n in names:
        print n

        image = imread(os.path.join(path, n))

        if len(image.shape) == 3 and image.shape[2] == 3:
            image = transform.resize(image, output_shape=(PIXELS, PIXELS, 3), preserve_range=True)
            image = image.transpose(2, 0, 1)
            train_image[i,:,:,:] = image

            if n.split('.')[0] == 'cat':
                levels[i] = 0
            else:
                levels[i] = 1

            i += 1

    return train_image, levels

train, levels = load_images(PATH)
train, levels = shuffle(train, levels)

print train.shape
print levels.shape

np.save('data/train_color.npy', train)
np.save('data/train_color_labels.npy', levels)

#for i in range(0,5):
#    j = randint(0, train.shape[0])
#    plot_sample(train[j])
#    print levels[j]

#print np.amax(train[0])
#print np.amin(train[0])

#print file_names
