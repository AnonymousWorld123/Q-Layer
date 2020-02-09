import numpy as np
import os
import sys
from six.moves import cPickle
import gzip
from keras.utils import np_utils

def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]
    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

def load_cifar(path = '../data/cifar/', resize = False):
    if resize == True:
        x_train = np.load('../data/cifar/resized_train.npy')
        x_test = np.load('../data/cifar/resized_test.npy')
        y_train = np.load('../data/cifar/train_label.npy')
        y_test = np.load('../data/cifar/test_label.npy')
    else:
        num_train_samples = 50000
        x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
        y_train = np.empty((num_train_samples,), dtype='uint8')
        for i in range(1, 6):
            fpath = os.path.join(path, 'data_batch_' + str(i))
            (x_train[(i - 1) * 10000: i * 10000, :, :, :],
            y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)
        fpath = os.path.join(path, 'test_batch')
        x_test, y_test = load_batch(fpath)
        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))
        # if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)
    return (x_train, y_train), (x_test, y_test)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels



from skimage import io
from skimage.color import gray2rgb
from skimage.transform import resize

def load_mnist(fashion = False, svhn_like = False):
    if fashion == True:
        path = '../data/fashion'
    else:
        path = '../data/mnist'
    X,Y = get_mnist(path, 'train')
    X2,Y2 = get_mnist(path, 't10k')
    if svhn_like == True:
        X_r = np.zeros((X.shape[0],32,32), np.uint8)
        X2_r = np.zeros((X2.shape[0],32,32), np.uint8)
        X_r[:,2:30,2:30] = X.reshape(-1,28,28)
        X2_r[:,2:30,2:30] = X2.reshape(-1,28,28)
        X = gray2rgb(X_r)
        X2 = gray2rgb(X2_r)
    return(X,Y,X2,Y2)


from scipy.io import loadmat

def load_svhn():
    path = '../data/svhn/'
    train = loadmat(path + 'train_32x32.mat')
    test = loadmat(path + 'test_32x32.mat')
    X = train['X'].transpose(-1,0,1,2)
    Y = train['y'][:,0]
    Y[Y==10] = 0
    X2 = test['X'].transpose(-1,0,1,2)
    Y2 = test['y'][:,0]
    Y2[Y2==10] = 0
    return(X, Y, X2, Y2)

def load_train_eval_test(name, fashion = False, norm = True):
    if name == 'cifar':
        (X, Y), (X2, Y2) = load_cifar(resize=False)

        x_train = X[:40000]
        x_test = X2
        x_val = X[40000:]
        x_sub = x_test[:150]

        y_train = np_utils.to_categorical(Y[:40000])
        y_test =  np_utils.to_categorical(Y2)
        y_val =  np_utils.to_categorical(Y[40000:])
        y_sub = y_test[:150]

        if norm:
            x_train = x_train/255
            x_val = x_val/255
            x_test = x_test/255

        _min = x_train.min()
        _max = x_train.max()

        return x_train, y_train, x_test, y_test, x_val, y_val, _min, _max
    
    elif name == 'mnist':

        X, Y, X2, Y2 = load_mnist(fashion=fashion)

        if norm:
            X = X/255
            X2 = X2/255

        x_train = X[:50000].reshape(50000, 28,28,1)
        x_test = X2[:].reshape(-1, 28,28,1)
        x_val = X[50000:].reshape(10000, 28,28,1)
        x_sub = X2[:150].reshape(150, 28,28,1)

        y_train = np_utils.to_categorical(Y[:50000])
        y_test =  np_utils.to_categorical(Y2[:])
        y_val =  np_utils.to_categorical(Y[50000:])
        y_sub = np_utils.to_categorical(Y2[:150])

        _min = x_train.min()
        _max = x_train.max()

        return x_train, y_train, x_test, y_test, x_val, y_val, _min, _max

    else:
        print('Wrong dataset name')
