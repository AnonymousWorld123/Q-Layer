import os
from skimage.io import imread
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np


class Utils(object):

    @staticmethod
    def image2kerasarray(img):
        ''' reshape image array 
        Args:
            img (numpy.ndarray)     : common image array ((nb_images,)height,width(,3))
        Returns:
            array (numpy.ndarray)   : keras array ((nb_images,)height,width,channels)
        '''
        array = img.astype('float32') / 255.
        if array.shape[-1] != 3: ### (num_images, height, width) ###
            array = np.expand_dims(array, axis=-1)

        return array


    @staticmethod
    def image2labelmap(img):
        ''' convert color image to label map
        Args:
            img (numpy.ndarray)     : common color image array (height,width,3)
        Returns:
            label_map (np.ndarray)	: label map (height*width*3,256)
        '''
        height, width, _  = img.shape
        label_map = np.zeros([height*width*3, 256])
        for h in range(height):
            for w in range(width):
                for c in range(3):
                    label_map[3*width*h + 3*w + c, img[h][w][c]] = 1

        return label_map


    @classmethod
    def load_mnist_datasets(
        cls,
        conditional=False):
        ''' load mnist dataset
        Args:
            conditional (bool)	: if True, return image arrays and class label vectors. if False, return only image arrays.
        Returns:
            ([X_train, h_train], Y_train), ([X_validation, h_validation], Y_validation)	: if conditional == True
            (X_train, Y_train), (X_validation, Y_validation)							: if conditional == False

        *** Loading {'cifar10', 'cifar100'} dataset causes MemoryError due to the softmax layer. ***

        '''
        from keras.datasets import mnist
        (X_train, h_train), (X_validation, h_validation) = mnist.load_data()
        nb_classes = 10

        from keras.utils import np_utils

        X_train = cls.image2kerasarray(X_train)
        X_validation = cls.image2kerasarray(X_validation)

        ### In case single channel, Y = X ###
        X_train = cls.binarize_array(X_train)
        X_validation = cls.binarize_array(X_validation)
        Y_train = X_train
        Y_validation = X_validation

        ### If conditional == True, use class labels as latent vector ###
        if conditional:
            h_train = np_utils.to_categorical(h_train, nb_classes)
            h_validation = np_utils.to_categorical(h_validation, nb_classes)


        if conditional:
            return (([X_train, h_train], Y_train), ([X_validation, h_validation], Y_validation))
        else:
            return ((X_train, Y_train), (X_validation, Y_validation))



    @classmethod
    def build_data_generator_from_keras_datasets(
        cls,
        dataset_name,
        X,
        H=None,
        batch_size=100):
        ''' kerasarray generator without keras ImageDataGenerator
            Args:
                dataset_name (str)	: {'mnist', 'cifar10', 'cifar100'}
                X (numpy.ndarray)	: Image arrays. (nb_images, height, width (, 3))
                H (numpy.ndarray)	: Latent vectors (nb_images, nb_classes)
                batch_size (int)	: minibatch size that generator yields at once
        '''
        from keras.utils import np_utils

        if dataset_name == 'mnist':
            target_size = (28, 28)
            nb_classes = 10
        elif dataset_name == 'cifar10':
            target_size = (32, 32)
            nb_classes = 10
        elif dataset_name == 'cifar100':
            target_size = (32, 32)
            nb_classes = 100

        while 1:
            if dataset_name == 'mnist':
                x = np.zeros((batch_size, target_size[0], target_size[1], 1))
                y = np.zeros((batch_size, target_size[0], target_size[1], 1))
            else:
                x = np.zeros((batch_size, target_size[0], target_size[1], 3))
                y = np.zeros((batch_size, target_size[0]*target_size[1]*3, 256))
            if H is not None:
                h = np.zeros((batch_size, nb_classes))
                H = np_utils.to_categorical(H, nb_classes)

            batch_idx = 0
            shuffled_index = list(range(len(X)))
            random.shuffle(shuffled_index)

            for i in shuffled_index:
                if dataset_name == 'mnist':
                    binarized_X = cls.binarize_array(cls.image2kerasarray(X[i]))
                    x[batch_idx % batch_size] = binarized_X
                    y[batch_idx % batch_size] = binarized_X
                else:
                    x[batch_idx % batch_size] = cls.image2kerasarray(X[i])
                    y[batch_idx % batch_size] = cls.image2labelmap(X[i])
                if h is not None:
                    h[batch_idx % batch_size] = H[i]
                batch_idx += 1
                if (batch_idx % batch_size) == 0:
                    if h is not None:
                        yield ([x, h], y)
                    else:
                        yield (x, y)


    @staticmethod
    def read_image(path):
        ''' read image from filepath
        Args:
            path (str)              : filepath (/path/to/img.jpg)
        Returns:
            image (numpy.ndarray)   : np.array of shepe (height, width, channels)
        '''
        img = imread(path)
        return img


    @classmethod
    def build_data_generator_from_directory(
        cls,
        target_size,
        data_paths,
        batch_size=100):
        ''' kerasarray generator without keras ImageDataGenerator
            Args:
                target_size (int,int)       : (height, width) pixels of image
                data_paths (list(str))      : ["/path/to/image001.jpg", "/path/to/image002.jpg", ...]
                batch_size (int)            : minibatch size that generator yields at once
        '''
        while 1:
            x = np.zeros((batch_size, target_size[0], target_size[1], 3))
            y = np.zeros((batch_size, target_size[0]*target_size[1]*3, 256))
            batch_idx = 0
            shuffled_index = list(range(len(data_paths)))
            random.shuffle(shuffled_index)

            for i in shuffled_index:
                x[batch_idx % batch_size] = cls.image2kerasarray(cls.read_image(data_paths[i]))
                y[batch_idx % batch_size] = cls.image2labelmap(cls.read_image(data_paths[i]))
                batch_idx += 1
                if (batch_idx % batch_size) == 0:
                    yield (x, y)
    

    @staticmethod
    def binarize_val(pred):
       ''' probability -> binarized value '''
       return (np.random.uniform(size=1) < pred).astype(np.float32)

    @staticmethod
    def binarize_array(array):
        ''' scaled image (value range:0-1) -> binarized array '''
        return (np.random.uniform(size=array.shape) < array).astype(np.float32)
    
    @staticmethod
    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        # copied from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
        preds = np.asarray(preds).astype('float32')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)

        return np.argmax(probas).astype('float32') / 255.


    @staticmethod
    def save_generated_image(
        img,
        filename="img.jpg",
        save_path='/tmp/pixelcnn/results'):
        ''' save predicted image
        Args:
            img (numpy.array)       : Image array
            filename (str)          : Save image to save_path/filename
            save_root (str)         : Save image to save_path/filename
        '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fig = plt.figure(figsize=(15, 15))
        ax = plt.subplot(1, 1, 1)
        ax.imshow(img)
        fig.savefig(os.path.join(save_path, filename))
        plt.close(fig)


    @staticmethod
    def save_generated_images(
        imgs,
        cols,
        filename='img.jpg',
        save_path='/tmp/pixelcnn/results'):
        ''' save predicted images
        Args:
            imgs (numpy.array)		: Image arrays
            cols (int)				: number of columns for visualizing images
            filename (str)			: Save image to save_path/filename
            save_root (str)			: Save image to save_path/filename
        '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fig = plt.figure()
        rows = len(imgs) // cols + 1

        for j in range(len(imgs)):
            ax = fig.add_subplot(cols, rows, j+1)
            ax.matshow(imgs[j], cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        plt.tight_layout()

        fig.savefig(os.path.join(save_path, filename))
        plt.close(fig)
