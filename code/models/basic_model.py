import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from sklearn.manifold import TSNE
from keras.models import Sequential
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta
from keras.activations import relu
from keras.models import Model
from keras.metrics import MSE, categorical_crossentropy
from keras.utils import np_utils
import random
import os  
import tensorflow as tf
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from models.vgg16_model import vgg_classifier, vgg_encoder
from keras import regularizers
from keras_radam import RAdam
from keras.regularizers import l2


# Load CNN models
def load_cnn(params, load_path, load_last = False):
    sess = tf.Session()
    G = sess.graph

    with sess.as_default():
        with G.as_default():
            cnn = CNN(sess=sess,**params)
            cnn.build()
            if load_last:
                cnn.load_best_lastpoint(load_path+'/')
            else:
                cnn.model.load_weights(load_path)

    return sess, cnn


def load_weights_from_source(target, source):
    for index, item in enumerate(target):
        weights = source[index].get_weights()
        if (len(weights)) == 0:
            continue
        item.set_weights(weights)
    print('Load weights done')




def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss

def get_encoder(mode='2d', **kwargs):
    print(mode)
    if (mode == '1d'):
        if 'kernel_size' in kwargs:
            kernel_size = kwargs['kernel_size']
            strides_size = kwargs['strides_size']
            filters = kwargs['filters']
        else:
            kernel_size = [8,4]
            strides_size = [4,2]
            filters = [16,32]
        encoder = Sequential()
        # encoder.add(Input(shape = kwargs['input_shape']))
        for i in range(len(kernel_size)):
            encoder.add(Conv1D(filters=filters[i],kernel_size=kernel_size[i], 
                    strides=strides_size[i], padding = 'same', name=str(i)+'_conv'))
            encoder.add(BatchNormalization())
            encoder.add(Activation(activation='relu'))
            encoder.add(MaxPooling1D(pool_size=2, name=str(i)+'_pooling'))  
    elif (mode == '2d'):
        encoder = Sequential()
        encoder.add(Conv2D(32, (3, 3), padding='same', input_shape = kwargs['input_shape'])) # receptive field 3, stride = 1
        encoder.add(Activation('relu'))
        encoder.add(Conv2D(64, (3, 3))) # receptive field 3 + 2 = 5, stride = 1
        encoder.add(Activation('relu'))
        encoder.add(MaxPooling2D(pool_size=(2, 2))) # receptive field 5 * 2 = 10, stride = 2
        encoder.add(Dropout(0.25))
        encoder.add(Conv2D(64, (3, 3), padding='same')) # receptive field 10 + 2*2 = 14, stride = 2
        encoder.add(Activation('relu'))
        encoder.add(Conv2D(64, (3, 3))) # receptive field 14 + 2*2  = 18
        encoder.add(Activation('relu'))
        # encoder.add(MaxPooling2D(pool_size=(2, 2))) # receptive field 36, stride = 4
        encoder.add(Dropout(0.25))
    elif 'vgg' in mode:
        encoder = get_inter_encoder(start = 'vgg16_start', end = mode, input_shape = kwargs['input_shape'])
    elif mode == 'A':
        encoder = Sequential()
        encoder.add(Conv2D(64, (5, 5), padding='same', input_shape = kwargs['input_shape']))
        encoder.add(Activation('relu'))
        encoder.add(Conv2D(filters = 64, kernel_size = (5, 5), strides = (2,2))) 
        encoder.add(Activation('relu'))
        encoder.add(Dropout(0.25))
    elif mode == 'B':
        encoder = Sequential()
        encoder.add(Dropout(0.25, input_shape = kwargs['input_shape']))
        encoder.add(Conv2D(filters = 64, kernel_size = (8, 8), strides = (2,2),padding='same', input_shape = kwargs['input_shape']))
        encoder.add(Activation('relu'))
        encoder.add(Conv2D(filters = 128, kernel_size = (6, 6), strides = (2,2))) 
        encoder.add(Activation('relu'))
        encoder.add(Conv2D(filters = 128, kernel_size = (5, 5), strides = (1,1))) 
        encoder.add(Activation('relu'))
        encoder.add(Dropout(0.5))
    elif mode == 'C':
        encoder = Sequential()
        encoder.add(Conv2D(filters = 128, kernel_size = (3, 3), padding='same', input_shape = kwargs['input_shape']))
        encoder.add(Activation('relu'))
        encoder.add(Conv2D(filters = 64, kernel_size = (3, 3), strides = (2,2))) 
        encoder.add(Activation('relu'))
        encoder.add(Dropout(0.25))
    elif mode == 'D':
        encoder = Sequential()
        encoder.add(Flatten(input_shape=kwargs['input_shape']))
        encoder.add(Dense(200, activation='relu',kernel_regularizer=regularizers.l2(0.0005)))
        encoder.add(Dropout(0.5))
    return(encoder)


def get_inter_encoder(start = 'vgg16_start', end = 'vgg16_pool1', input_shape = (32,32,3), weight_decay = 0.0005):
    model = Sequential()
    if start == 'vgg16_start':

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        if end == 'vgg16_pool0.5':
            return model

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block1_pool'))

        if end == 'vgg16_pool1':
            return model

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block2_pool'))

        if end == 'vgg16_pool2':
            return model

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block3_pool'))

        if end == 'vgg16_pool3':
            return model

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block4_pool'))

        if end == 'vgg16_pool4':
            return model

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block5_pool'))
        model.add(Dropout(0.5))

    if start == 'vgg16_pool0.5':
        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay),input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block1_pool'))

        if end == 'vgg16_pool1':
            return model
        
        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block2_pool'))
        
        if end == 'vgg16_pool2':
            return model

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block3_pool'))

        if end == 'vgg16_pool3':
            return model

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block4_pool'))

        if end == 'vgg16_pool4':
            return model

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block5_pool'))
        model.add(Dropout(0.5))
    

    if start == 'vgg16_pool1':
        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), input_shape = input_shape))
    
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block2_pool'))

        if end == 'vgg16_pool2':
            return model

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block3_pool'))

        if end == 'vgg16_pool3':
            return model

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block4_pool'))

        if end == 'vgg16_pool4':
            return model

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block5_pool'))
        model.add(Dropout(0.5))
    
    if start == 'vgg16_pool2':

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block3_pool'))

        if end == 'vgg16_pool3':
            return model

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block4_pool'))

        if end == 'vgg16_pool4':
            return model

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block5_pool'))
        model.add(Dropout(0.5))
    
    if start == 'vgg16_pool3':

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block4_pool'))

        if end == 'vgg16_pool4':
            return model

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block5_pool'))
        model.add(Dropout(0.5))
    
    if start == 'vgg16_pool4':

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2), name = 'block5_pool'))
        model.add(Dropout(0.5))

    if start == 'A_start':
        model.add(Conv2D(64, (5, 5), padding='same', input_shape = input_shape))
        model.add(Activation('relu'))
        return(model)
    if start == 'A_inter':
        model.add(Conv2D(filters = 64, kernel_size = (5, 5), strides = (2,2),input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
    return(model)


def get_classifier(pre_class = 10, classifier_dim = 64, regularizer = 0.0005, BN = None, faltten = True):
    classifier= Sequential()
    if faltten == True:
        classifier.add(Flatten())
    if (classifier_dim != 0):
        classifier.add(Dense(classifier_dim,use_bias=True,kernel_regularizer=regularizers.l2(regularizer)))
        classifier.add(ReLU())
    # classifier.add(BatchNormalization())
    classifier.add(Dropout(0.5))
    classifier.add(Dense(pre_class,use_bias=True))
    classifier.add(Softmax())
    return(classifier)

class CNN():
    def __init__(self, sess, lr, pre_class, mode, classifier_dim, input_shape, regularizer, save_path, optimizer, **kwarg):
        self.sess = sess
        self.lr = lr
        self.pre_class = pre_class
        self.mode = mode
        self.classifier_dim = classifier_dim
        self.input_shape = input_shape
        self.regularizer = regularizer
        self.save_path = save_path
        self.optimizer = optimizer
        if os.path.exists(self.save_path) == False:
            os.mkdir(self.save_path)
            print(self.save_path)
        # self.save_params()
    
    def build(self):
        self.encoder = get_encoder(self.mode, input_shape = self.input_shape)
        self.input_x = Input(shape = self.input_shape)
        self.e = self.encoder(self.input_x)
        if len(self.e.shape) < 4:
            self.classifier = get_classifier(self.pre_class, self.classifier_dim, regularizer=self.regularizer, faltten = False)
            self.pre = self.classifier(self.e)  
        else: 
            self.classifier = get_classifier(self.pre_class, self.classifier_dim, regularizer=self.regularizer)
            self.pre = self.classifier(self.e)   

        self.model = Model(self.input_x, self.pre)
        self.model.compile(loss='categorical_crossentropy', optimizer = eval(self.optimizer)(lr=self.lr), 
                    metrics=['acc'])

    def build_trans(self, pre_class, lr = None):
        if (lr == None):
            lr = self.lr
        input_x = self.model.layers[0].input
        self.trans_y_true = tf.placeholder(tf.float32,[None, pre_class])
        self.trans_classifier = get_classifier(pre_class, self.classifier_dim)
        self.trans_pre = self.trans_classifier(self.model.layers[1](input_x))
        self.trans_classifier.layers[1].set_weights(self.model.layers[-1].layers[1].get_weights())
        self.trans_model =  Model(input_x, self.trans_pre)
        self.trans_model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=lr), metrics=['accuracy'])
    
    def train(self, train_X, train_Y, val_X = None, val_Y = None, TRAIN_EPOCH = 15, BATCH_SIZE = 32, verbose = 2, trans = False):
        if trans == False:
            model = self.model
        else:
            model = self.trans_model
        print('Training CNN')
        if val_X is not None:
            hist = model.fit(train_X,train_Y, batch_size=BATCH_SIZE, epochs=TRAIN_EPOCH, verbose=verbose, validation_data=(val_X, val_Y))
        else:
            hist = model.fit(train_X,train_Y, batch_size=BATCH_SIZE, epochs=TRAIN_EPOCH, verbose=verbose)
        print('Finish training CNN')
        return(hist)

    def save(self, save_path = './save/',trans = False):
        if (trans == False):
            self.model.save_weights(save_path + '_model')
        else:
            self.trans_model.save_weights(save_path + '_trans_model')
    
    def load(self, save_path = './save/',trans=False):
        if (trans == False):
            self.model.load_weights(save_path + '_model')
        else:
            self.trans_model.load_weights(save_path + '_trans_model')
    
    def save_encoder(self, save_path = './save/'):
        self.encoder.save(save_path + '_encoder')
    
    def load_last_checkpoint(self, path = None):
        if path is None:
            path = self.save_path

        l_ = os.listdir(path)

        l = []
        for item in l_:
            if 'ckpt' in item:
                l.append(item)

        latest = sorted(l)[-1]
        print('Load',path + latest)

        self.model.load_weights(path + latest)

    def find_best(self, x_val, y_val, path = None):
        if path is None:
            path = self.save_path
        paths = sorted(os.listdir(path))

        best_acc = -1
        best_path = paths[0]
        for p in paths:
            self.model.load_weights(path + p)
            acc = self.model.evaluate(x_val, y_val)[-1]
            print(p,acc)
            if acc > best_acc:
                best_acc = acc
                best_path = p
        
        f = open(path + 'best_model','w')
        d = f.write(best_path)
        f.close()
        print(best_path)
    
    def load_best_lastpoint(self, path = None):
        if path is None:
            path = self.save_path
        
        file_path = path + 'best_model'
        if os.path.exists(file_path):
            best_path = open(path + 'best_model','r').readlines()[0]
            self.model.load_weights(path + '/' + best_path)
            print(best_path)
        else:
            self.load_last_checkpoint(path)





def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape = (32,32,3), depth=20, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model





