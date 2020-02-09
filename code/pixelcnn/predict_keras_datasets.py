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

from keras.datasets import mnist
from keras.utils import np_utils
from keras.utils.visualize_util import plot


def predict(argv=None):
    ''' generate image from conditional Gated PixelCNN model 
    Usage:
    	python predict_keras_datasets.py -c sample_predict.cfg         : prediction example using configfile
    	python predict_keras_datasets.py --option1 hoge ...            : predict with command-line options
        python predict_keras_datasets.py -c predict.cfg --opt1 hoge... : overwrite config options with command-line options
    '''

    ### parsing arguments from command-line or config-file ###
    if argv is None:
        argv = sys.argv

    conf_parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
        )
    conf_parser.add_argument("-c", "--conf_file",
                        help="Specify config file", metavar="FILE_PATH")
    args, remaining_argv = conf_parser.parse_known_args()

    defaults = {}

    if args.conf_file:
        config = configparser.SafeConfigParser()
        config.read([args.conf_file])
        defaults.update(dict(config.items("General")))

    parser = argparse.ArgumentParser(
        parents=[conf_parser]
        )
    parser.set_defaults(**defaults)
    parser.add_argument("--checkpoint_file", help="Checkpoint file [Required]", type=str, metavar="FILE_PATH")
    parser.add_argument("--conditional", help="model the conditional distribution p(x|h) (default:False)", type=str, metavar="BOOL")
    parser.add_argument("--dataset_name", help="{'mnist','cifar10','cifar100'}", type=str, metavar="DATASET_NAME")
    parser.add_argument("--class_label", help="Class label (default: 0)",type=int, metavar="INT")
    parser.add_argument("--nb_images", help="Number of images to generate",type=int, metavar="INT")
    parser.add_argument("--batch_size", help="Batch size at prediction",type=int, metavar="INT")
    parser.add_argument("--temperature", help="Temparature value for sampling diverse values (default: 1.0)",type=float, metavar="FLOAT")
    parser.add_argument("--nb_pixelcnn_layers", help="Number of PixelCNN Layers (except last two ReLu layers)",type=int, metavar="INT")
    parser.add_argument("--nb_filters", help="Number of filters for each layer",type=int, metavar="INT")
    parser.add_argument("--filter_size_1st", help="Filter size for the first layer. (default: (7,7))", metavar="INT,INT")
    parser.add_argument("--filter_size", help="Filter size for the subsequent layers. (default: (3,3))", metavar="INT,INT")
    parser.add_argument("--save_path", help="Root directory which generated images are saved (default: /tmp/pixelcnn/results)", type=str, metavar="DIR_PATH")

    args = parser.parse_args(remaining_argv)

    utils = Utils()

    try:
        checkpoint_file = args.checkpoint_file
        dataset_name = args.dataset_name
    except ValueError:
        sys.exit("Error: --checkpoint_file must be specified.")

    conditional = strtobool(args.conditional) if args.conditional else False
    temperature = args.temperature if args.temperature else 1.0

    ### mnist image size ###
    if dataset_name == 'mnist':
        input_size = (28, 28)
        nb_classes = 10
        nb_channels = 1
    elif dataset_name == 'cifar10':
        input_size = (32, 32)
        nb_classes = 10
        nb_channels = 3
    elif dataset_name == 'cifar100':
        input_size = (32, 32)
        nb_classes = 100
        nb_channels = 3

    ### build PixelCNN model ###
    model_params = {}
    model_params['input_size'] = input_size
    model_params['nb_channels'] = nb_channels 
    model_params['conditional'] = conditional
    model_params['latent_dim'] = nb_classes
    if args.nb_pixelcnn_layers:
        model_params['nb_pixelcnn_layers'] = int(args.nb_pixelcnn_layers)
    if args.nb_filters:
        model_params['nb_filters'] = int(args.nb_filters)
    if args.filter_size_1st:
        model_params['filter_size_1st'] = tuple(map(int, args.filter_size_1st.split(',')))
    if args.filter_size:
        model_params['filter_size'] = tuple(map(int, args.filter_size.split(',')))

    
    save_path = args.save_path if args.save_path else '/tmp/pixelcnn/results'

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    pixelcnn = PixelCNN(**model_params)
    pixelcnn.build_model()
    pixelcnn.model.load_weights(checkpoint_file)


    ## prepare zeros array
    class_label = int(args.class_label) if args.class_label else 0
    nb_images = int(args.nb_images) if args.nb_images else 8
    batch_size = int(args.batch_size) if args.batch_size else nb_images
    if dataset_name == 'mnist':
        X_pred = np.zeros((nb_images, input_size[0], input_size[1], 1))
    else:
        X_pred = np.zeros((nb_images, input_size[0], input_size[1], 3))
    if conditional:
        h_pred = np_utils.to_categorical(class_label, nb_classes)
        h_pred = np.repeat(h_pred, nb_images, axis=0)
    
    ### generate images pixel by pixel
    for i in range(input_size[0]):
        for j in range(input_size[1]):
            for k in range(nb_channels):
                if conditional:
                    x = [X_pred, h_pred]
                else:
                    x = X_pred

                next_X_pred = pixelcnn.model.predict(x, batch_size)
                if dataset_name == 'mnist':
                    binarizer = lambda x: utils.binarize_val(x)
                    binarized_pred = np.vectorize(binarizer)(next_X_pred[:,i,j,k])
                    X_pred[:,i,j,k] = binarized_pred
                else:
                    sampled_pred = next_X_pred[:,i*input_size[1]*nb_channels+j*nb_channels+k,:]
                    sampled_pred = np.array([utils.sample(sampled_pred[i]) for i in range(len(sampled_pred))])
                    X_pred[:,i,j,k] = sampled_pred

    X_pred = np.squeeze(X_pred)
    X_pred = (255*X_pred).astype(np.uint8)

    ### save images ###
    for i in range(nb_images):
        utils.save_generated_image(X_pred[i], 'generated_'+str(i)+'.jpg', save_path)

    return (0)


if __name__ == '__main__':
    sys.exit(predict())
