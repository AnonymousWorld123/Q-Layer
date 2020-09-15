import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)

import keras.backend as K
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.utils import np_utils
import os
from load_data import *
from models.vq_models import *
from utils import *
from params import cifar_param, mnist_param, fashion_param
from art.attacks import DeepFool, FastGradientMethod, BasicIterativeMethod, BoundaryAttack
from art.classifiers import KerasClassifier, TFClassifier
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)







def lr_schedule(epoch, lr):
    if (epoch > 0) and (epoch % 10 ==0):
        lr = 0.5 * lr
    return lr


reduce_lr =  LearningRateScheduler(lr_schedule, verbose = 1)
earlyStopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='auto')


def rawtrain_cnn(param, x_train, y_train, x_val, y_val, epochs=100, datagen = None, batch_size = 128, save_best = True):
    # train CNN
    sess = tf.Session()
    G = sess.graph

    with sess.as_default():
        with G.as_default():
            cnn = CNN(sess=sess, **param)
            cnn.build()
            learning_rate = param['lr']
            if save_best != False:
                save_best = True
            mcp_save = ModelCheckpoint(cnn.save_path+'{epoch:03d}-{acc:03f}-{val_acc:03f}.ckpt', save_best_only=save_best, period = 1, monitor='val_acc', mode='auto',verbose=1)
            cnn.model.compile(loss='categorical_crossentropy', optimizer = eval(param['optimizer'])(lr=learning_rate),  metrics=['acc'])
            if datagen is None:
                cnn.model.fit(x_train, y_train, batch_size=param['batch_size'],callbacks=[reduce_lr,earlyStopping,mcp_save], validation_data=[x_val,y_val], epochs = epochs, shuffle=True)
            else:
                cnn.model.fit_generator(
                    datagen.flow(x_train, y_train, batch_size=batch_size),
                    callbacks=[reduce_lr,earlyStopping,mcp_save], validation_data=[x_val,y_val], epochs = epochs , 
                    steps_per_epoch = x_train.shape[0]//batch_size, shuffle=True)

            cnn.load_last_checkpoint()
    
    return sess, cnn



def generate_adv_samples(classifier, sess, attack, x_val):
    with sess.as_default():
        with sess.graph.as_default():
            adv_crafter = attack(classifier, batch_size = 128)
            x_val_adv = adv_crafter.generate(x_val)
    
    return x_val_adv


def rawtrain_vq(params, config, x_train, y_train, x_val, y_val, epochs = 100, datagen = None, batch_size = 128, load = None):
    print(params['save_path'])

    if ('inter_layer' in params) and (params['inter_layer'] is not None):
        vq_hard = VQ_clustering_inter(config, params) 
    else:
        vq_hard = VQ_clustering(config, params) 

    if load is None:
        vq_hard.init_embeds_flag = 'kmeans'
    
    else:
        vq_hard.init_embeds_flag = False
        vq_hard.load_lastpoint(load)

    if datagen is None:
        loss_c, loss_vq, loss_btclass, loss_inactive, loss, accs_q, accs_e, avg = vq_hard.train(TRAIN_EPOCH = epochs,
            train_X = x_train, train_Y = y_train, val_X = x_val, val_Y = y_val, BATCH_SIZE = batch_size, summary_it = 100)
    
    else:
        loss_c, loss_vq, loss_btclass, loss_inactive, loss, accs_q, accs_e, avg = vq_hard.train(TRAIN_EPOCH = epochs,
            train_X = x_train, train_Y = y_train, val_X = x_val, val_Y = y_val, BATCH_SIZE = batch_size, summary_it = 100,
            train_flow=datagen.flow(x_train, y_train, batch_size=batch_size), step_per_epoch=x_train.shape[0]//batch_size)
    
    vq_hard.load_lastpoint()
    return vq_hard


def MNIST_CNN_1(**args):
    '''
    Train a regular CNN for generating a validation set and adversarial testing sets.
    '''

    # Params
    if args['fashion'] == False:
        print('Train MNIST')
        params = mnist_param
    else:
        print('Train Fashion')
        params = fashion_param
    
    # Get data
    x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test('mnist', fashion = args['fashion'])

    for item in args:
        if args[item] is not None:
            params.cnn[item] = args[item]
    
    if args['save_path'] is not None:
        params.cnn['save_path'] = params.save_path + args['save_path']
    else:
        params.cnn['save_path'] = params.save_path + 'cnn_source/'

    print('Save path:', params.cnn['save_path'])

    # Rawtrain CNN
    sess_cnn, cnn = rawtrain_cnn(params.cnn, x_train, y_train, x_val, y_val, epochs = args['epochs'], save_best = args['save_best'])
    classifier_cnn = KerasClassifier(model=cnn.model, clip_values=(_min, _max))
    

    # Generate adv_val for selection
    # attack = FastGradientMethod
    # kwargs = {
    #     'eps':args['eps'],
    #     'batch_size': 128,
    #     'eps_step':0.1,
    # }

    # x_val_adv = generate_adv_samples(classifier_cnn, sess_cnn, attack, x_val)

    # if args['fashion'] == False:
    #     np.save('./save_val/mnist_x_val_adv_vgg.npy',x_val_adv)
    # else:
    #     np.save('./save_val/fashion_x_val_adv_vgg.npy',x_val_adv)


def MNIST_CNN_2(**args):
    '''
    Add x_val_adv to val set
    Train a baseline CNN (use to compare with the VQ model)
    '''

    # Params
    if args['fashion'] == False:
        params = mnist_param
    else:
        params = fashion_param

    # Get data
    x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test('mnist', fashion = args['fashion'])

    # Add adv samples to the val set
    if args['fashion'] == False:
        x_val_adv = np.load('./save_val/mnist_x_val_adv_vgg.npy')
    else:
        x_val_adv = np.load('./save_val/fashion_x_val_adv_vgg.npy')

    x_val = np.append(x_val, x_val_adv, axis=0)
    y_val = np.append(y_val, y_val, axis=0)

    for item in args:
        if args[item] is not None:
            params.cnn[item] = args[item]
    
    # Rawtrain CNN
    if args['save_path'] is None:
        params.cnn['save_path'] = params.save_path + 'cnn_baseline/'
    else:
        params.cnn['save_path'] = params.save_path + args['save_path']
    
    sess_cnn, cnn = rawtrain_cnn(params.cnn, x_train, y_train, x_val, y_val, epochs = args['epochs'], save_best = args['save_best'])


def MNIST_VQ(**args):

    # Params
    if args['fashion'] == False:
        params = mnist_param
    else:
        params = fashion_param

    save_path = params.save_path
    if os.path.exists(save_path + 'hard/') == False:
        os.mkdir(save_path + 'hard/')


    # Get data
    x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test('mnist', fashion = args['fashion'])

    # Add adv samples to the val set
    if args['fashion'] == False:
        x_val_adv = np.load('./save_val/mnist_x_val_adv_vgg.npy')
    else:
        x_val_adv = np.load('./save_val/fashion_x_val_adv_vgg.npy')

    x_val = np.append(x_val, x_val_adv, axis=0)
    y_val = np.append(y_val, y_val, axis=0)

    hard_save_path = save_path + 'hard/'
    # Rawtrain VQ
    params_hard = np.copy(params.hard_inter).item()

    for item in args:
        if args[item] is not None:
            params_hard[item] = args[item]
        
    space_mode = 'trans_' if params_hard['set_trans'] > params_hard['set_fixed'] else 'fixed_'
    num_space = max(params_hard['set_trans'], params_hard['set_fixed'])

    params_hard['concept_dim'] = args['dim'] // num_space
    params_hard['dense_dim'] = args['dim'] // num_space

    params_hard['save_path'] = hard_save_path + space_mode + str(params_hard['inter_layer']) +'_' +str(num_space) + '_' + str(params_hard['num_concept']) + '_' + str(params_hard['c1']) + '_' + str(params_hard['c2'])+ '_' + str(params_hard['alpha']) + '_' + str(params_hard['beta']) + '_lr_' + str(params_hard['lr'])+ '_dim_'+str(params_hard['concept_dim']) + '_share_' +str(params_hard['share_backward']) + '_update_'+str(params_hard['update'])  +'/'
    
    i = 0
    while os.path.exists(params_hard['save_path']):
        params_hard['save_path'] = params_hard['save_path'][:-1] + '_' + str(i) + '/'

    vq_hard = rawtrain_vq(params_hard,config, x_train, y_train, x_val, y_val, epochs = args['epochs'])


def CIFAR_CNN_1(**args):


    # Params
    params = cifar_param
    if args['save_path'] is not None:
        params.cnn['save_path'] = params.save_path + args['save_path']

    # Get data for cifar
    x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test('cifar')

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

    # Rawtrain CNN
    sess_cnn, cnn = rawtrain_cnn(params.cnn_pool3, x_train, y_train, x_val, y_val, datagen=datagen)
    classifier_cnn = KerasClassifier(model=cnn.model, clip_values=(_min, _max))

    # Generate adv_val for selection
    # attack = FastGradientMethod
    # kwargs = {
    #     'eps':args['eps'],
    #     'batch_size': 128,
    #     'eps_step':0.1,
    # }

    # x_val_adv = generate_adv_samples(classifier_cnn, sess_cnn, attack, x_val)

    # # Add adv samples to the val set
    # x_val_adv = np.load('./save_val/cifar_01_pool3_x_val_adv.npy')
    # x_val = np.append(x_val, x_val_adv, axis=0)
    # y_val = np.append(y_val, y_val, axis=0)


def CIFAR_CNN_2(**args):
    # Params
    params = cifar_param
    save_path = params.save_path

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

    # Rawtrain CNN
    if args['save_path'] is None:
        params.cnn_pool3['save_path'] = save_path + 'cnn_pool3_baseline/'
    else:
        params.cnn_pool3['save_path'] = save_path + args['save_path']

    sess_cnn, cnn = rawtrain_cnn(params.cnn_pool3, x_train, y_train, x_val, y_val, datagen=datagen)
    classifier_cnn = KerasClassifier(model=cnn.model, clip_values=(_min, _max))


def CIFAR_VQ(**args):
    params = cifar_param
    save_path = params.save_path
    
    if os.path.exists(save_path + 'hard/') == False:
        os.mkdir(save_path + 'hard/')

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

    hard_save_path = save_path
    params_hard = np.copy(params.hard_pool3).item()

    # Rawtrain VQ
    for item in args:
        if args[item] is not None:
            params_hard[item] = args[item]
        
    space_mode = 'trans_' if params_hard['set_trans'] > params_hard['set_fixed'] else 'fixed_'
    num_space = max(params_hard['set_trans'], params_hard['set_fixed'])

    params_hard['concept_dim'] = args['dim'] // num_space
    params_hard['dense_dim'] = args['dim'] // num_space

    params_hard['save_path'] = hard_save_path + space_mode + str(params_hard['inter_layer']) +'_' +str(num_space) + '_' + str(params_hard['num_concept']) + '_' + str(params_hard['c1']) + '_' + str(params_hard['c2'])+ '_' + str(params_hard['alpha']) + '_' + str(params_hard['beta']) + '_lr_' + str(params_hard['lr'])+ '_dim_'+str(params_hard['concept_dim']) + '_share_' +str(params_hard['share_backward']) + '_update_'+str(params_hard['update'])  +'/'
    
    i = 0
    while os.path.exists(params_hard['save_path']):
        params_hard['save_path'] = params_hard['save_path'][:-1] + '_' + str(i) + '/'

    vq_hard = rawtrain_vq(params_hard,config, x_train, y_train, x_val, y_val, datagen = datagen, epochs= args['epochs'])



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help='run which task',default='MNIST_VQ', required=False)
    parser.add_argument("--fashion", type=str2bool, help='fashion mnist?',default='False')

    parser.add_argument("--eps", type=float, help='eps used to generate validation data?',default=0.3)
    parser.add_argument("--epochs",type=int, help='update', default=100)
    parser.add_argument("--save_path",type=str, help='save path', required=False)
    parser.add_argument("--save_best",type=str, help='save_best', default='1')

    parser.add_argument("--num_concept",type=int, help='num_concept',required=False)
    parser.add_argument("--set_trans",type=int, help='set_trans',default=0) # These are different projection methods
    parser.add_argument("--set_att",type=int, help='set_fixed',default=0) # These are different projection methods
    parser.add_argument("--set_fixed",type=int, help='set_fixed',default=1) # These are different projection methods
    parser.add_argument("--dim",type=int, help='dense_dim',default=64) 
    # parser.add_argument("--concept_dim",type=int, help='concept_dim',default=16)
    # parser.add_argument("--classifier_dim",type=int, help='concept_dim',required=False)

    parser.add_argument("--mode",type=str, help='mode', required=False)
    parser.add_argument("--start_layer",type=str, help='start_layer', required=False)
    parser.add_argument("--inter_layer",type=str, help='inter_layer', required=False)
    parser.add_argument("--share_backward",type=str2bool, help='share_backward', default='True')
    parser.add_argument("--update",type=str, help='update', default='train')

    parser.add_argument("--early_stop",type=int, help='early_stop', required = False)

    parser.add_argument("--c1",type=float, help='c1', required = False)
    parser.add_argument("--c2",type=float, help='c2', required = False)
    parser.add_argument("--alpha",type=float, help='alpha', required = False)

    parser.add_argument("--gpu",type=str, help='GPU', default='0', required=False)

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
    
    if args.update == 'False':
        args.update = False

    args = vars(args)
    args['beta'] = args['alpha']
    eval(args['task'])(**args)

if __name__ == "__main__":
    main()