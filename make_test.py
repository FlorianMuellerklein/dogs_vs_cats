import os
import numpy as np
import pandas as pd
import cPickle as pickle
from natsort import natsorted
from random import randint

from skimage import exposure
from matplotlib import pyplot
from skimage.io import imread
from PIL import Image
from skimage.io import imshow
from skimage.filters import sobel
from skimage import feature
from skimage.color import gray2rgb

from sklearn.preprocessing import StandardScaler

PATH = '/Volumes/Mildred/Kaggle/DogsvsCats/data/test1'

maxPixel = 192
imageSize = maxPixel * maxPixel
num_features = imageSize * 3

def plot_sample(x):
    img = x.reshape(maxPixel, maxPixel, 3)
    imshow(img)
    pyplot.show()

def load_images(path):
    print 'reading file names ... '
    names = [d for d in os.listdir (path) if d.endswith('.jpg')]
    names = natsorted(names)
    num_rows = len(names)
    print names

    print 'making dataset ... '
    test_image = np.zeros((num_rows, num_features), dtype = float)
    label = np.zeros((num_rows, 1), dtype = int)
    file_names = []
    i = 0
    for n in names:
        print n.split('.')[0]

        image = imread(os.path.join(path, n))
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image.transpose(2, 0, 1)
            test_image[i, 0:num_features] = np.reshape(image, (1, num_features))
            label[i] = n.split('.')[0]
            i += 1
        else:
            image = gray2rgb(image)
            image = image.transpose(2, 0, 1)
            test_image[i, 0:num_features] = np.reshape(image, (1, num_features))
            label[i] = n.split('.')[0]
            i += 1

    return test_image, label

test, label = load_images(PATH)

print test[0]
print test.shape

np.save('data/test_color.npy', np.hstack((test, label)))

for i in range(0,5):
    j = randint(0, test.shape[0])
    plot_sample(test[j])
    print label[j]

print np.amax(test[0])
print np.amin(test[0])

#print file_names
