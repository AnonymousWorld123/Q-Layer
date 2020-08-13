import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import keras.backend as K
from art.classifiers import KerasClassifier, TFClassifier
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.utils import np_utils
import os
from models.vq_models import *
from models.basic_model import *
import argparse
from utils import *
import logging
from load_data import *
from params import cifar_pool5_regularizer, mnist_A, fashion_mnist_A
from pixelcnn.core.layers import PixelCNN
import seaborn as sb


# random seed
seed_value = 0
np.random.seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)

# Params

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Tools
def get_zq(X, vq_hard, reshape = False):
    with vq_hard.sess.as_default():
        with vq_hard.sess.graph.as_default():
            feature_model = Model(vq_hard.input_x, vq_hard.z_q)
            features = feature_model.predict(X, batch_size=256)
    return features

def get_zk(X, vq_hard):
    with vq_hard.sess.as_default():
        with vq_hard.sess.graph.as_default():
            feature_model = Model(vq_hard.input_x, vq_hard.z_k)
            features = feature_model.predict(X, batch_size=256)

    temp = features
    temp = np_utils.to_categorical(np.uint8(temp), num_classes = vq_hard.num_concept, dtype=np.uint8).transpose(1,2,0, -1).reshape(X.shape[0],-1,vq_hard.num_concept)
    return temp

def get_ze(X, vq_hard):
    with vq_hard.sess.as_default():
        with vq_hard.sess.graph.as_default():
            feature_model = Model(vq_hard.input_x, vq_hard.z_e)
            features = feature_model.predict(X, batch_size=256)
    return features


def get_cnn_feature(X, cnn):
    with cnn.sess.as_default():
        with cnn.sess.graph.as_default():
            feature_model = Model(cnn.model.layers[1].layers[0].input, cnn.model.layers[1].layers[3].output)
            features = feature_model.predict(X, batch_size=256)
    
    return features

def get_data(X, vq_hard, batch_size = 256):
    indexs = list(range(len(X)))
    shuffle(indexs)
    step = 0
    while True:
        X_q_train = get_zq(X[indexs][step : step+ batch_size], vq_hard)
        X_k_train = get_zk(X[indexs][step : step+ batch_size], vq_hard)
        step = step + batch_size
        if step > len(X):
            step = 0
            shuffle(indexs)
        yield X_q_train, X_k_train



def get_pixel_data(X, batch_size = 128):
    indexs = list(range(len(X)))
    shuffle(indexs)
    step = 0
    while True:
        X_q_train = X[indexs][step : step+ batch_size]
        X_k_train = np_utils.to_categorical(X_q_train, num_classes=256, dtype='uint8')
        if len(X_k_train.shape) == 5:
            X_k_train = X_k_train.reshape(X_k_train.shape[:-2] + (-1,))
        step = step + batch_size
        if step > len(X):
            step = 0
            shuffle(indexs)
        yield X_q_train, X_k_train
    


def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))




# Train and evaluate
def vq_train_pixelcnn(pixelcnn_params, vq_hard, x_train, x_q_val, y_k_val, epochs):

    if not os.path.exists(pixelcnn_params['save_root']):
        os.makedirs(pixelcnn_params['save_root'])
    
    json_str = json.dumps(pixelcnn_params)
    with open(pixelcnn_params['save_root'] + '/params.json', 'w') as json_file:
        json_file.write(json_str)

    pixelcnn = PixelCNN(**pixelcnn_params)
    pixelcnn.build_model()
    reduce_lr =  ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='min', min_lr=1e-6)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    mcp_save = ModelCheckpoint(pixelcnn_params['save_root']+'/pixelcnn-weights.{epoch:02d}-{val_loss:.4f}.ckpt', save_best_only=False, period=1, monitor='val_accuracy', mode='max',verbose=1, save_weights_only=True)
    pixelcnn.model.compile(loss='categorical_crossentropy', optimizer = RAdam(0.001), metrics = ['accuracy',f1])
    pixelcnn.model.fit_generator(get_data(x_train, vq_hard, batch_size=batch_size), steps_per_epoch = len(x_train) // batch_size, epochs = epochs, validation_data = (x_q_val, y_k_val), callbacks=[reduce_lr, earlyStopping, mcp_save])
    pixelcnn.model.evaluate(x_q_val, y_k_val)

    return pixelcnn


def pixel_train_pixelcnn(pixelcnn_params, x_train, x_val, epochs):

    if not os.path.exists(pixelcnn_params['save_root']):
        os.makedirs(pixelcnn_params['save_root'])
    
    json_str = json.dumps(pixelcnn_params)
    with open(pixelcnn_params['save_root'] + '/params.json', 'w') as json_file:
        json_file.write(json_str)

    pixelcnn = PixelCNN(**pixelcnn_params)
    pixelcnn.build_model()
    reduce_lr =  ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='min', min_lr=1e-6)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    mcp_save = ModelCheckpoint(pixelcnn_params['save_root']+'/pixelcnn-weights.{epoch:02d}-{val_loss:.4f}.ckpt', save_best_only=False, period=1, monitor='val_accuracy', mode='max',verbose=1, save_weights_only=True)
    
    x_val_one_hot = np_utils.to_categorical(x_val, 256, dtype='uint8').reshape(x_val.shape[:-1] + (-1,))
    pixelcnn.model.compile(loss='categorical_crossentropy', optimizer = RAdam(0.001), metrics = ['accuracy',f1])
    pixelcnn.model.fit_generator(get_pixel_data(x_train, batch_size=batch_size), steps_per_epoch = len(x_train) // batch_size, epochs = epochs, validation_data = (x_val, x_val_one_hot), callbacks=[reduce_lr, earlyStopping, mcp_save])

    return pixelcnn

def load_pixelcnn(path):
    f = open(path + '/params.json')
    params = json.load(f)
    params['save_root'] = path

    # find last checkpoint
    l_ = os.listdir(path)
    l = []
    for item in l_:
        if 'ckpt' in item:
            l.append(item)

    latest = sorted(l)[-1]
    print('Load',path + latest)

    pixelcnn = PixelCNN(**params)
    pixelcnn.build_model()
    pixelcnn.model.load_weights(params['save_root']+ '/' + latest)

    return pixelcnn


def pixelcnn_evaluate(pixelcnn, attack_file_path, x_test, num_classes = 256):
    x_test_one_hot = np_utils.to_categorical(x_test, num_classes, dtype='uint8').reshape(x_test.shape[:-1] + (-1,))

    x_test_advs = np.load(attack_file_path, allow_pickle=True).item()
    adv_pixel_logs = {}
    if 0 not in x_test_advs:
        x_test_advs[0] = x_test

    print('---Pixel PixelCNN---')
    for key in x_test_advs:
        adv_out_pixel = pixelcnn.model.predict(x_test_advs[key], batch_size=batch_size)
        adv_out_pixel_log = np.sum(np.log(adv_out_pixel)*x_test_one_hot, axis=-1)
        adv_out_pixel_log_sum = np.sum(adv_out_pixel_log.reshape(x_test_advs[key].shape[0],-1), axis=1)
        adv_pixel_logs[key] = adv_out_pixel_log_sum

        # print(key, adv_out_pixel_log_sum)
    
    return adv_pixel_logs

def vq_pixelcnn_evaluate(vq_hard, pixelcnn, attack_file_path, x_test):
    x_test_advs = np.load(attack_file_path, allow_pickle=True).item()
    adv_pixel_logs = {}
    if 0 not in x_test_advs:
        x_test_advs[0] = x_test

    print('---VQ PixelCNN---')
    for key in x_test_advs:
        x_q_test = get_zq(x_test_advs[key], vq_hard)
        y_k_test = get_zk(x_test_advs[key], vq_hard)
        adv_out_pixel = pixelcnn.model.predict(x_q_test, batch_size=batch_size)
        adv_out_pixel_log = np.sum(np.log(adv_out_pixel)*y_k_test, axis=-1)
        adv_out_pixel_log_sum = np.sum(adv_out_pixel_log.reshape(x_q_test.shape[0],-1), axis=1)
        adv_pixel_logs[key] = adv_out_pixel_log_sum

        # print(key, adv_out_pixel_log_sum)
    
    return adv_pixel_logs
    



def VQ_train(**args):  
    # Get val data
    x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test(args['data'], fashion = args['fashion'])

    # Params
    if args['data'] == 'cifar':
        params = cifar_pool5_regularizer
    else:
        if args['fashion'] == False:
            params = mnist_A
        else:
            params = fashion_mnist_A

    save_path = params.save_path

    # Load VQ
    path = args['load_path']
    vq_hard = load_vq(path, config, args['last_name'])

    # Train Pixel CNN

    x_q_val = get_zq(x_val, vq_hard)
    y_k_val = get_zk(x_val, vq_hard)

    pixelcnn_params = {
    'input_size': x_q_val.shape[1:],
    'nb_channels':x_q_val.shape[-1],
    'conditional':False,
    'latent_dim':10,
    'nb_pixelcnn_layers':args['layers'],
    'nb_filters':args['filters'],
    'filter_size_1st':(5,5),
    'filter_size':(3,3),
    'optimizer':'adam',
    'es_patience':1,
    'save_best_only':True,
    'num_concept': vq_hard.params['num_concept'],
    'num_spaces':vq_hard.params['set_fixed'],
    }

    pixelcnn_params['save_root'] = vq_hard.params['save_path'] + '/pixelcnn_' + str(args['layers']) + '_' + str(args['filters']) + '/'
    i = 0
    while os.path.exists(pixelcnn_params['save_root']):
        pixelcnn_params['save_root'] = pixelcnn_params['save_root'][:-1] + '_' + str(i) + '/'
        i = i+1

    pixelcnn = vq_train_pixelcnn(pixelcnn_params, vq_hard, x_train, x_q_val, y_k_val, epochs = args['epochs'])            


def pixel_train(**args):
    # Get val data
    print(args)
    x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test(args['data'], fashion = args['fashion'], norm = False)

    # Params
    if args['data'] == 'cifar':
        params = cifar_pool5_regularizer
    else:
        if args['fashion'] == False:
            params = mnist_A
        else:
            params = fashion_mnist_A

    save_path = params.save_path

    # Load VQ
    pixelcnn_params = {
        'input_size': x_train.shape[1:],
        'nb_channels':x_train.shape[-1],
        'conditional':False,
        'latent_dim':10,
        'nb_pixelcnn_layers':args['layers'],
        'nb_filters':args['filters'],
        'filter_size_1st':(5,5),
        'filter_size':(3,3),
        'optimizer':'adam',
        'es_patience':100,
        'save_best_only':True,
        'num_concept':256,
        'num_spaces':x_train.shape[-1]
    }

    pixelcnn_params['save_root'] = save_path+ '/pixelcnn_' + str(args['layers']) + '_' + str(args['filters']) + '/'
    i = 0
    while os.path.exists(pixelcnn_params['save_root']):
        pixelcnn_params['save_root'] = pixelcnn_params['save_root'][:-1] + '_' + str(i) + '/'
        i = i + 1

    pixelcnn = pixel_train_pixelcnn(pixelcnn_params, x_train, x_val, epochs = args['epochs'])


import scipy.spatial.distance as distance

def cosine(x, y):
    return (x*y).sum(-1) / np.sqrt((x*x).sum(-1) * (y*y).sum(-1))

def cnn_distance_evaluate(cnn, attack_file_path, x_test, gaussian = False, dimensions = [0,64]):
    x_test_advs = np.load(attack_file_path, allow_pickle=True).item()
    adv_distances = {}
    x_q_test_clean = get_cnn_feature(x_test, cnn)
    x_q_test_clean = x_q_test_clean.reshape(-1, x_q_test_clean.shape[-1])[:,dimensions[0]:dimensions[1]]
    for key in x_test_advs:
        if gaussian == False:
            x_test_adv = x_test_advs[key]
        else:
            x_test_adv = x_test + np.random.normal(scale = key, size = x_test.shape)
        x_q_test = get_cnn_feature(x_test_advs[key], cnn)
        x_q_test = x_q_test.reshape(-1, x_q_test.shape[-1])[:,dimensions[0]:dimensions[1]]
        distance = cosine(x_q_test_clean, x_q_test)
        adv_distances[key] = distance
    #
    return adv_distances

def vq_distance_evaluate(vq_hard, attack_file_path, x_test, path='q', gaussian = False, dimensions = [0,64]):
    x_test_advs = np.load(attack_file_path, allow_pickle=True).item()
    adv_distances = {}
    #
    if path == 'q':
        feature_func = get_zq
    else:
        feature_func = get_ze
    # 
    x_q_test_clean = feature_func(x_test, vq_hard)
    x_q_test_clean = x_q_test_clean.reshape(-1, x_q_test_clean.shape[-1])[:,dimensions[0]:dimensions[1]]
    for key in x_test_advs:
        if gaussian == False:
            x_test_adv = x_test_advs[key]
        else:
            x_test_adv = x_test + np.random.normal(scale = key, size = x_test.shape)
        x_q_test = feature_func(x_test_adv, vq_hard)
        x_q_test = x_q_test.reshape(-1, x_q_test.shape[-1])[:,dimensions[0]:dimensions[1]]
        distance = cosine(x_q_test_clean, x_q_test)
        adv_distances[key] = distance
    #
    return adv_distances


def evaluate_distance(**args):
    # Get data
    x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test(args['data'], fashion = args['fashion'], norm = True)

    # Load attack file and evaluate
    if args['data'] == 'cifar':
        params = cifar_pool5_regularizer
        paths = ['save_val/attacks/CIFAR_pool3_BIM_copy_x_test_advs.npy', 'save_val/attacks/CIFAR_pool3_FGSM_copy_x_test_advs.npy']
    else:
        if args['fashion'] == False:
            name = 'MNIST'
            params = mnist_A
        else:
            name = 'Fashion'
            params = fashion_mnist_A

        paths = ['save_val/attacks/'+name+ '_BIM_copy_x_test_advs.npy', 'save_val/attacks/'+name+ '_FGSM_copy_x_test_advs.npy']


    save_path = params.save_path

    # Load CNN and VQ
    sess_cnn, cnn = load_cnn(cifar_pool5_regularizer.cnn, args['cnn_path'], load_last=True)            
    vq_hard = load_vq(args['load_path'], config, args['last_name'])


    attack_names = ['BIM', 'FGSM']
    for index, black_flag in enumerate([args['BIM_black'], args['FGSM_black']]):
        if black_flag:
            attack_file_path = paths[index]

            # Get distance logs
            cnn_attack_distance = cnn_distance_evaluate(cnn, attack_file_path, x_test, gaussian=False, dimensions = [0,64])
            vq_attack_distance_e = vq_distance_evaluate(vq_hard, attack_file_path, x_test, path = 'e', gaussian=False, dimensions = [0,64])
            vq_attack_distance_q = vq_distance_evaluate(vq_hard, attack_file_path, x_test, path = 'q', gaussian=False, dimensions = [0,64])

            print('---CNN Distance---')
            for key in cnn_attack_distance:
                print(key, cnn_attack_distance[key].mean())

            print('---VQ E-Path Distance---')
            for key in vq_attack_distance_e:
                print(key, vq_attack_distance_e[key].mean())

            print('---VQ Q-Path Distance---')
            for key in vq_attack_distance_q:
                print(key, vq_attack_distance_q[key].mean())
                    



def plot_distance_figure(cnn_attack_distance, vq_attack_distance_e, vq_attack_distance_q, key):
    cmap = plt.get_cmap('tab10')
    results = [
        vq_attack_distance_q[key],
        vq_attack_distance_e[key],
        cnn_attack_distance[key]
    ]
    
    name_list = ['z_q','z_e', 'CNN']
    maps = [3,1,0]
    f, ax =  plt.subplots(nrows=1, ncols=1, figsize=(10,5))

    for i in range(3):
        c = results[i]
        align = 'mid'
        n, bins, patches = ax.hist(x=c, bins=10, range=(0,1), color=cmap(maps[i]), label=name_list[i],
                                        alpha=0.4, align=align, weights=np.ones(len(c)) / len(c))

    plt.legend()
    f.savefig('./distance_figure',bbox_inches='tight',dpi=f.dpi,pad_inches=0.0)

def evaluate(**args):
    # Get data
    if args['mode'] == 'pixel':
        norm = False
    else:
        norm = True

    x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test(args['data'], fashion = args['fashion'], norm = norm)

    # Load Pixel CNN

    # Load attack file and evaluate
    if args['data'] == 'cifar':
        params = cifar_pool5_regularizer
        paths = ['save_val/attacks/CIFAR_pool3_BIM_copy_x_test_advs.npy', 'save_val/attacks/CIFAR_pool3_FGSM_copy_x_test_advs.npy']
    else:
        if args['fashion'] == False:
            name = 'MNIST'
            params = mnist_A
        else:
            name = 'Fashion'
            params = fashion_mnist_A

        paths = ['save_val/attacks/'+name+ '_BIM_copy_x_test_advs.npy', 'save_val/attacks/'+name+ '_FGSM_copy_x_test_advs.npy']


    save_path = params.save_path

    attack_names = ['BIM', 'FGSM']
    for index, black_flag in enumerate([args['BIM_black'], args['FGSM_black']]):
        if black_flag:
            attack_file_path = paths[index]

            # Get attack logs
            if args['mode'] == 'pixel':
                pixelcnn = load_pixelcnn(args['pixelcnn_path'])
                attack_logs = pixelcnn_evaluate(pixelcnn, attack_file_path, x_test, num_classes = 256)
            else:
                vq_hard = load_vq(args['load_path'], config, args['last_name'])
                with vq_hard.sess.as_default():
                    with vq_hard.sess.graph.as_default():
                        pixelcnn = load_pixelcnn(args['pixelcnn_path'])
                        attack_logs = vq_pixelcnn_evaluate(vq_hard, pixelcnn, attack_file_path, x_test)

            clean_log = attack_logs[0]


            # Plot figure
            for key in attack_logs:
                if key != 0:
                    attack_log = attack_logs[key]
                    save_path = args['pixelcnn_path'] + '/' + attack_names[index] + '_' + str(key) + '.png'
                    size = pixelcnn.input_size[0]
                    num_classes = pixelcnn.num_concept * pixelcnn.num_spaces
                    plot_figures(clean_log, attack_log, save_path, size, num_classes)
                    kl = KL(clean_log/size/size/num_classes, attack_log/size/size/num_classes)
                    print(attack_names[index] + '_' + str(key), kl)


                

def plot_figures(test_out_log_sum, adv_out_log_sum, save_path, size = 28, num_classes = 256):
    f = plt.figure(figsize=(10,8))
    plt.rcParams.update({'font.size': 20})
    sb.distplot(test_out_log_sum/size/size/num_classes,label='clean')
    sb.distplot(adv_out_log_sum/size/size/num_classes,label='adv')
    plt.legend()
    plt.title('Distribution')
    f.savefig(save_path,bbox_inches='tight',dpi=f.dpi,pad_inches=0.0)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help='run which task', required=True)
    parser.add_argument("--mode", type=str, help='VQ or pixel', required=False)

    parser.add_argument("--data", type=str, help='mnist or cifar?',default='mnist')
    parser.add_argument("--fashion", type=bool, help='fashion mnist?',default=False)
    parser.add_argument("--batch_size", type=int, help='batch_size',default=128)

    parser.add_argument("--layers",type=int, help='layers for pixelcnn', default=7)
    parser.add_argument("--filters",type=int, help='filters for pixelcnn', default=64)
    parser.add_argument("--epochs", type=int, help='epochs for training pixelcnn',default=50)


    parser.add_argument("--load_path",type=str, help='vq path', required=False)
    parser.add_argument("--last_name",type=str, help='load_last', required=False)

    parser.add_argument("--pixelcnn_path",type=str, help='pixelcnn_path', required=False)

    parser.add_argument("--gpu",type=str, help='GPU', default='0', required=False)

    parser.add_argument("--FGSM_black",type=bool, help='FGSM black-box attack', default=False)
    parser.add_argument("--BIM_black",type=bool, help='BIM black-box attack', default=False)
    parser.add_argument("--FGSM_black_large",type=bool, help='FGSM large black-box attack', default=False)
    parser.add_argument("--BIM_black_large",type=bool, help='BIM large black-box attack', default=False)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    global batch_size
    batch_size = args.batch_size

    global config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    K.set_session(tf.Session(config=config))

    args = vars(args)
    eval(args['task'])(**args)


if __name__ == '__main__':
    # MNIST_generate_black_box_data()
    main()
