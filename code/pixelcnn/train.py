import os
import sys
import argparse
import configparser
from datetime import datetime
import pytz
from distutils.util import strtobool
import numpy as np

from core.layers import PixelCNN
from core.utils import Utils

from keras.utils.visualize_util import plot


def train(x_train, y_train, x_val, y_val, model_params):
    '''
    model_params should include
    ("--nb_epoch", help="Number of epochs [Required]", type=int, metavar="INT")
    ("--batch_size", help="Minibatch size", type=int, metavar="INT")
    ("--conditional", help="model the conditional distribution p(x|h) (default:False)", type=str, metavar="BOOL")
    ("--nb_pixelcnn_layers", help="Number of PixelCNN Layers (exept last two ReLu layers)", metavar="INT")
    ("--nb_filters", help="Number of filters for each layer", metavar="INT")
    ("--filter_size_1st", help="Filter size for the first layer. (default: (7,7))", metavar="INT,INT")
    ("--filter_size", help="Filter size for the subsequent layers. (default: (3,3))", metavar="INT,INT")
    ("--optimizer", help="SGD optimizer (default: adadelta)", type=str, metavar="OPT_NAME")
    ("--es_patience", help="Patience parameter for EarlyStopping", type=int, metavar="INT")
    ("--save_root", help="Root directory which trained files are saved (default: /tmp/pixelcnn)", type=str, metavar="DIR_PATH")
    ("--timezone", help="Trained files are saved in save_root/YYYYMMDDHHMMSS/ (default: Asia/Tokyo)", type=str, metavar="REGION_NAME")
    ("--save_best_only", help="The latest best model will not be overwritten (default: False)", type=str, metavar="BOOL")
    input_shape
    nb_classes
    '''

    if not os.path.exists(model_params['save_root']):
        os.makedirs(model_params['save_root'])


    pixelcnn = PixelCNN(**model_params)
    pixelcnn.build_model()
    pixelcnn.model.summary()

    train_params = {}
    train_params['x'] = x_train
    train_params['y'] = y_train
    train_params['validation_data'] = (x_val, y_val)
    train_params['nb_epoch'] = nb_epoch
    train_params['batch_size'] = batch_size
    train_params['shuffle'] = True

    pixelcnn.fit(**train_params)

    return (pixelcnn)

