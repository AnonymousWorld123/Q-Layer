import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.utils import np_utils
import os
from load_data import *
from models.vq_models import *
from models.basic_model import *
from utils import *
from params import cifar_pool5_regularizer, mnist_A, fashion_mnist_A
from art.attacks import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent
from art.classifiers import KerasClassifier, TFClassifier
from art.defences import *
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# random seed
seed_value = 0
np.random.seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)


def lr_schedule(epoch, lr):
    if (epoch > 0) and (epoch % 20 ==0):
        lr = 0.5 * lr
    return lr


reduce_lr =  LearningRateScheduler(lr_schedule, verbose = 1)
earlyStopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='auto')



'''Training'''


def datagen(x_train, y_train, batch_size):
    indexs = list(range(len(x_train)))
    shuffle(indexs)
    step = 0
    while True:
        batch_xs = x_train[indexs][step : step+ batch_size]
        batch_ys = y_train[indexs][step : step+ batch_size]
        step = step + batch_size
        print(step)

        if step > len(x_train):
            step = 0
            shuffle(indexs)

        yield batch_xs, batch_ys

def gen_generator(gen, attack, sess, rate = 1):
    attack_ratio = rate
    while True:
        batch_xs, batch_ys = next(gen)
        nb_adv = int(np.ceil(attack_ratio * batch_xs.shape[0]))

        if attack_ratio < 1:
            adv_ids = np.random.choice(batch_xs.shape[0], size=nb_adv, replace=False)
        else:
            adv_ids = list(range(batch_xs.shape[0]))
            # np.random.shuffle(adv_ids)

        with sess.as_default():
            with sess.graph.as_default():
                batch_xs[adv_ids] = attack.generate(batch_xs[adv_ids])
                yield batch_xs, batch_ys


def adv_train_cnn(cnn, sess, attack, save_path, x_train, y_train, x_val, y_val, ratio = 1, epochs = 20, batch_size = 128, datagen = datagen):

    if os.path.exists(save_path) == False:
        os.mkdir(save_path)

    with sess.as_default():
        with sess.graph.as_default():

            gen = gen_generator(
                datagen(x_train, y_train, batch_size=batch_size),
                attack = attack, rate = ratio, sess = sess
                )
            mcp_save = ModelCheckpoint(save_path +'{epoch:03d}-{acc:03f}-{val_acc:03f}.ckpt', save_best_only=False, period = 1,monitor='val_acc', mode='auto',verbose=1)
            cnn.model.fit_generator(
                gen, 
                callbacks=[reduce_lr,earlyStopping,mcp_save], validation_data=[x_val,y_val], epochs = epochs , 
                steps_per_epoch = x_train.shape[0]//batch_size, shuffle=True)
            
            cnn.load_last_checkpoint()

            # print('Test acc:',cnn.model.evaluate(x_test,y_test))

    return cnn



def adv_train_vq(vq_hard, attacks, x_train, y_train, x_val, y_val, ratio = 1, epochs = 20, batch_size = 128, datagen = None):
    '''
    paht: load pretrained vq model
    '''
    vq_hard.reset_global_step()
    vq_hard.save_params()
    if datagen is None:
        loss_c, loss_vq, loss_btclass, loss_inactive, loss, accs_q, accs_e, avg = vq_hard.train(TRAIN_EPOCH = epochs,
            train_X = x_train, train_Y = y_train, val_X = x_val, val_Y = y_val, BATCH_SIZE = batch_size, summary_it = 100,
            attacks=attacks, attack_ratio = ratio)
    else:
        loss_c, loss_vq, loss_btclass, loss_inactive, loss, accs_q, accs_e, avg = vq_hard.train(TRAIN_EPOCH = epochs,
                train_X = x_train, train_Y = y_train, val_X = x_val, val_Y = y_val, BATCH_SIZE = batch_size, summary_it = 100,
                train_flow=datagen.flow(x_train, y_train, batch_size=batch_size), step_per_epoch=x_train.shape[0]//batch_size,
                attacks = attacks, attack_ratio = ratio)
    # print('Test E acc/ Q acc :',vq_hard.evaluate(x_test, y_test)[-3:-1])

    return vq_hard






def MNIST_CNN(**args):

    # Get val data
    x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test('mnist', fashion = args['fashion'])
    if args['fashion'] == False:
        x_val_adv = np.load('./save_val/mnist_x_val_adv_vgg.npy')
    else:
        x_val_adv = np.load('./save_val/fashion_x_val_adv_vgg.npy')
    
    x_val = np.append(x_val, x_val_adv, axis=0)
    y_val = np.append(y_val, y_val, axis=0)

    # Params
    if args['fashion'] == False:
        params = mnist_A
    else:
        params = fashion_mnist_A

    save_path = params.save_path

    # Adv train CNN
    sess_cnn, cnn = load_cnn(params.cnn,args['load_path'], load_last=True)
    classifier_cnn = KerasClassifier(model=cnn.model, clip_values=(_min, _max))


    if args['PGD'] == False:
        cnn_attack = FastGradientMethod(eps=args['eps'], eps_step = 0.01, batch_size=128, classifier = classifier_cnn)
    else:
        cnn_attack = ProjectedGradientDescent(eps=args['eps'], eps_step = 0.01, batch_size=128, classifier = classifier_cnn)
    
    cnn_adv_save = args['load_path']  + '/adv_eps_' + str(args['eps']) +  '_ratio_' + str(args['ratio']) + '_PGD_' + str(args['PGD']) +'/'

    if os.path.exists(cnn_adv_save) == False:
        os.mkdir(cnn_adv_save)
    else:
        i = 0
        while os.path.exists(cnn_adv_save):
            cnn_adv_save = cnn_adv_save[:-1] + '_' + str(i) + '/'
    
        os.mkdir(cnn_adv_save)
    
    print('Save at',cnn_adv_save)
    
    cnn = adv_train_cnn(cnn, sess_cnn, cnn_attack, cnn_adv_save, x_train, y_train, x_val, y_val, epochs=args['epochs'], ratio=args['ratio'])


def MNIST_VQ(**args):
    # Get val data
    x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test('mnist', fashion = args['fashion'])
    
    if args['fashion'] == False:
        x_val_adv = np.load('./save_val/mnist_x_val_adv_vgg.npy')
    else:
        x_val_adv = np.load('./save_val/fashion_x_val_adv_vgg.npy')
    
    x_val = np.append(x_val, x_val_adv, axis=0)
    y_val = np.append(y_val, y_val, axis=0)

    # Params
    if args['fashion'] == False:
        params = mnist_A
    else:
        params = fashion_mnist_A

    save_path = params.save_path

    # Adv train VQ
    batch_size = 128

    path = args['load_path']
    vq_hard = load_vq(path, config, args['last_name'])
    classifer_e = TFClassifier(input_ph=vq_hard.input_x,  output = vq_hard.pre_e, labels_ph= vq_hard.y_true,clip_values=(_min, _max), sess=vq_hard.sess, loss=vq_hard.loss_c2)
    
    if args['PGD'] == False:
        vq_attacks = [FastGradientMethod(classifer_e, eps = args['eps'], batch_size = batch_size)]
    else:
        vq_attacks = [ProjectedGradientDescent(classifer_e, eps = args['eps'], eps_step = 0.01, batch_size = batch_size)]
    
    vq_hard.params['save_path'] = path + '/adv_eps_' + str(args['eps']) +  '_ratio_' + str(args['ratio'])  + '_PGD_' + str(args['PGD']) + '/'   

    if os.path.exists(vq_hard.params['save_path']) == False:
        os.mkdir(vq_hard.params['save_path'])

    else:
        i = 0

        while os.path.exists(vq_hard.params['save_path']):
            vq_hard.params['save_path'] = vq_hard.params['save_path'][:-1] + '_' + str(i) + '/'

        os.mkdir(vq_hard.params['save_path'])
    
    print('Save at',vq_hard.params['save_path'])

        
    vq_hard.params['save_best'] = 'q' if args['save_best'] is None else args['save_best']
    vq_hard.params['early_stop'] = args['early_stop']

    adv_train_vq(vq_hard, vq_attacks, x_train, y_train, x_val, y_val, ratio=args['ratio'], epochs=args['epochs'])


def CIFAR_CNN(**args):
    # Get data for cifar
    x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test('cifar')
    x_val_adv = np.load('./save_val/cifar_01_pool3_x_val_adv.npy')
    x_val = np.append(x_val, x_val_adv, axis=0)
    y_val = np.append(y_val, y_val, axis=0)

    datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)

    # Params
    params = cifar_pool5_regularizer
    save_path = params.save_path

    # Adv train CNN
    sess_cnn, cnn = load_cnn(params.cnn_pool3, args['load_path'], load_last= True)
    classifier_cnn = KerasClassifier(model=cnn.model, clip_values=(_min, _max))

    cnn_adv_save = args['load_path'] + '/adv_eps_' + str(args['eps']) +  '_ratio_' + str(args['ratio'])  + '_PGD_' + str(args['PGD']) + '/'   
    if os.path.exists(cnn_adv_save) == False:
        os.mkdir(cnn_adv_save)

    else:
        i = 0
        while os.path.exists(cnn_adv_save):
            cnn_adv_save = cnn_adv_save[:-1] + '_' + str(i) + '/'

        os.mkdir(cnn_adv_save)

    print('Save at',cnn_adv_save)

    if args['PGD'] == False:
        cnn_attack = FastGradientMethod(eps=args['eps'], eps_step = 0.01, batch_size=128, classifier = classifier_cnn)
    else:
        cnn_attack = ProjectedGradientDescent(eps=args['eps'], eps_step = 0.01, batch_size=128, classifier = classifier_cnn)
    
    cnn = adv_train_cnn(cnn, sess_cnn, cnn_attack, cnn_adv_save, x_train, y_train, x_val, y_val, datagen=datagen.flow, epochs=100, ratio=args['ratio'])

def CIFAR_VQ(**args):
    # Get data for cifar
    x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test('cifar')
    x_val_adv = np.load('./save_val/cifar_01_pool3_x_val_adv.npy')
    x_val = np.append(x_val, x_val_adv, axis=0)
    y_val = np.append(y_val, y_val, axis=0)

    datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)

    # Params
    params = cifar_pool5_regularizer
    save_path = params.save_path

    # Adv train VQ
    batch_size = 128

    path = args['load_path']
    vq_hard = load_vq(path, config, args['last_name'])
    vq_hard.build_path()
    classifer_e = TFClassifier(input_ph=vq_hard.input_x,  output = vq_hard.pre_e, labels_ph= vq_hard.y_true,clip_values=(_min, _max), sess=vq_hard.sess, loss=vq_hard.loss_c2)
    classifer_q_path = TFClassifier(input_ph=vq_hard.input_x,  output = vq_hard.pre_path, labels_ph= vq_hard.y_true,clip_values=(_min, _max), sess=vq_hard.sess, loss=vq_hard.loss_path)

    if args['PGD'] == False:
        vq_attacks = [FastGradientMethod(classifer_e, eps = args['eps'], batch_size = batch_size)]
    else:
        vq_attacks = [ProjectedGradientDescent(classifer_e, eps = args['eps'], eps_step = 0.01, batch_size = batch_size)]
    # eps = 0.02
    # vq_attacks = [FastGradientMethod(classifer_e, eps = eps, batch_size = batch_size),FastGradientMethod(classifer_q_path, eps = eps, batch_size = batch_size)]
    
    vq_hard.params['save_path'] = path + '/adv_eps_' + str(args['eps']) +  '_ratio_' + str(args['ratio'])  + '_PGD_' + str(args['PGD']) + '/'   
    
    if os.path.exists(vq_hard.params['save_path']) == False:
        os.mkdir(vq_hard.params['save_path'])

    else:
        i = 0

        while os.path.exists(vq_hard.params['save_path']):
            vq_hard.params['save_path'] = vq_hard.params['save_path'][:-1] + '_' + str(i) + '/'
        
        os.mkdir(vq_hard.params['save_path'])

    print('Save at',vq_hard.params['save_path'])

    vq_hard.params['save_best'] = 'q' if args['save_best'] is None else args['save_best']
    vq_hard.params['early_stop'] = args['early_stop']

    adv_train_vq(vq_hard, vq_attacks, x_train, y_train, x_val, y_val, ratio=args['ratio'], epochs=args['epochs'], datagen = datagen)



def main():   
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help='run which task',default='CIFAR_CNN', required=False)
    parser.add_argument("--fashion", type=bool, help='fashion mnist?',default=False)

    parser.add_argument("--epochs",type=int, help='update', default= 100)
    parser.add_argument("--early_stop",type=int, help='early_stop', default= 30)

    parser.add_argument("--load_path",type=str, help='load path', required=False, default='save_val/cifar_01/cnn_pool3_baseline/')
    parser.add_argument("--last_name",type=str, help='load_last', required=False)

    parser.add_argument("--save_best",type=str, help='save_best', default='q', required=False)
    parser.add_argument("--gpu",type=str, help='GPU', default='0', required=False)

    parser.add_argument("--eps",type=float, help='eps', default=0.2, required=False)
    parser.add_argument("--ratio",type=float, help='ratio', default=1, required=False)
    parser.add_argument("--PGD", type=bool, help='PGD training?',default=False)



    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    global config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    K.set_session(tf.Session(config=config))

    if args.save_best.isdigit():
        args.save_best = int(args.save_best)
        
    args = vars(args)
    eval(args['task'])(**args)

if __name__ == "__main__":
    main()
