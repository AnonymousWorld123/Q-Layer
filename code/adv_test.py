import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)

import keras.backend as K
from art.attacks import *
from art.classifiers import KerasClassifier, TFClassifier
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from art.defences import *
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
from params import cifar_param, mnist_param, fashion_param

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




# # Generate adversarial samples
def get_adv_samples(attack, eps_list, x_test, classifier, sess,step = 0.01):
    kwargs = {
        'batch_size': 128,
        'eps_step':step,
    }
    x_test_advs = {}
    with sess.as_default():
        with sess.graph.as_default():
            tf.set_random_seed(seed_value)
            for eps in eps_list:
                kwargs['eps'] = eps
                adv_crafter = attack(classifier, **kwargs)
                x_test_adv = adv_crafter.generate(x_test)
                x_test_advs[eps] = x_test_adv
                # acc = cnn_copy.model.evaluate(x_test_adv, y_test)
                # print(acc)
                # preds = np.argmax(classifier.predict(x_test_adv, batch_size=32), axis=1)
                # acc = np.sum(preds == np.argmax(y_test, axis=1)) / len(x_test_adv)
                # logger.info('Accuracy on adversarial samples: %.2f%%', (acc * 100))
    return(x_test_advs)




'''Test'''
# Black box
def vq_black_box_test(attack_file_path, models, y_test, x_test):
    x_test_advs = np.load(attack_file_path, allow_pickle=True).item()
    if 0 not in x_test_advs:
        x_test_advs[0] = x_test
    vq_accs_q, vq_accs_e = get_vq_accs(models, x_test_advs,y_test)

    return vq_accs_q, vq_accs_e

# CNN
def cnn_black_box_test(attack_file_path, models, y_test, x_test):
    x_test_advs = np.load(attack_file_path, allow_pickle=True).item()
    if 0 not in x_test_advs:
        x_test_advs[0] = x_test
    accs = get_cnn_accs(models, x_test_advs, y_test)

    return accs



# White box

def vq_white_box_test(epss, attacks, sesses, clfs, models, names, x_test, y_test):
    for attack in attacks:

        for eps in epss:
            print(attack, eps)

            if attack == BasicIterativeMethod:
                kwargs = {
                    'eps':eps, # For gradient based
                    'batch_size': 128,
                    'eps_step': 0.01,
                }
            
            elif attack == FastGradientMethod:
                kwargs = {
                    'eps':eps, # For gradient based
                    'batch_size': 128,
                    'eps_step': eps,
                }
            
            else:
                kwargs = {
                    'eps':eps, # For gradient based
                    'batch_size': 128,
                }
            
            for index,classifier in enumerate(clfs):
                with sesses[index].as_default():
                    with sesses[index].graph.as_default():
                        logger.info('-'*30)
                        logger.info(names[index])

                        adv_crafter = attack(classifier, **kwargs)
                        x_test_adv = adv_crafter.generate(x_test)
                        acc = models[index].evaluate(x_test_adv, y_test)[-3:-1]
                        acc_q = round(acc[0]*100,2)
                        acc_e = round(acc[1]*100,2)

                        logger.info('q/e ' + str(acc_q) + '/' + str(acc_e)) 

    


def cnn_white_box_test(epss, attacks, sesses, clfs, models, names, x_test, y_test):
    for attack in attacks:
        for eps in epss:
            print(attack, eps)

            if attack == BasicIterativeMethod:
                kwargs = {
                    'eps':eps, # For gradient based
                    'batch_size': 128,
                    'eps_step': 0.01,
                }
            
            elif attack == FastGradientMethod:
                kwargs = {
                    'eps':eps, # For gradient based
                    'batch_size': 128,
                    'eps_step': eps,
                }
            
            else:
                kwargs = {
                    'eps':eps, # For gradient based
                    'batch_size': 128,
                }

            for index,classifier in enumerate(clfs):
                with sesses[index].as_default():
                    with sesses[index].graph.as_default():
                        logger.info('-'*30)
                        logger.info(names[index])

                        adv_crafter = attack(classifier, **kwargs)
                        x_test_adv = adv_crafter.generate(x_test)
                        preds = np.argmax(classifier.predict(x_test_adv, batch_size=32), axis=1)
                        acc = np.sum(preds == np.argmax(y_test, axis=1)) / len(x_test_adv)

                        logger.info('Accuracy on adversarial samples: %.2f%%', (acc * 100))
    



def CIFAR_generate_black_box_data(**args):
    x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test('cifar')

    # Params
    params = cifar_param
    save_path = params.save_path

    # Load CNN
    params.cnn_pool3['save_path'] = params.save_path + args['cnn_path']
    sess_cnn, cnn = load_cnn(params.cnn_pool3, params.cnn_pool3['save_path'], load_last=True)
    classifier_cnn = KerasClassifier(model=cnn.model, clip_values=(_min, _max))

    # Generate Black box data
    eps_list = eval(args['eps'])
    classifier = classifier_cnn
    sess = sess_cnn

    attack_name = ['FGSM','BIM']
    if args['validation_file'] == True:
        attack = FastGradientMethod
        x_test_advs = get_adv_samples(attack, eps_list, x_test, classifier, sess)
        np.save('save_val/cifar_01_pool3_x_val_adv.npy', x_test_advs[eps_list[0]])
    else:
        for index, attack in enumerate([FastGradientMethod, BasicIterativeMethod]):
            x_test_advs = get_adv_samples(attack, eps_list, x_test, classifier, sess)
            x_test_advs[0] = x_test
            np.save('save_val/attacks/CIFAR_pool3_' + attack_name[index] + '_copy_x_test_advs.npy',x_test_advs)




def CIFAR(**args):

    # Get data for cifar
    x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test('cifar')

    # Params
    params = cifar_param
    save_path = params.save_path

    # Load CNN
    if args['cnn_path'] is not None:
        if args['last_name'] is not None:
            sess_cnn, cnn = load_cnn(params.cnn, args['cnn_path'] + args['last_name'], load_last=False)
        else:
            sess_cnn, cnn = load_cnn(params.cnn, args['cnn_path'], load_last=True)
        classifier_cnn = KerasClassifier(model=cnn.model, clip_values=(_min, _max))

    # Load VQ
    if args['vq_path'] is not None:
        path = args['vq_path']
        vq_hard = load_vq(path, config, args['last_name'])
        classifer_e = TFClassifier(input_ph=vq_hard.input_x,  output = vq_hard.pre_e, labels_ph= vq_hard.y_true,clip_values=(_min, _max), sess=vq_hard.sess, loss=vq_hard.loss_c2)
        vq_models = [vq_hard]
    
    # Black box test
    paths = ['save_val/attacks/CIFAR_pool3_BIM_copy_x_test_advs.npy', 'save_val/attacks/CIFAR_pool3_BIM_large_copy_x_test_advs.npy',\
        'save_val/attacks/CIFAR_pool3_FGSM_copy_x_test_advs.npy', 'save_val/attacks/CIFAR_pool3_FGSM_large_copy_x_test_advs.npy']
    
    for index, black_flag in enumerate([args['BIM_black'], args['BIM_black_large'], args['FGSM_black'], args['FGSM_black_large']]):
        if black_flag:
            print(paths[index])

            attack_file_path = paths[index]
            if args['cnn_path'] is not None:
                accs = cnn_black_box_test(attack_file_path, cnn, y_test, x_test)
            if args['vq_path'] is not None:
                vq_accs_q, vq_accs_e = vq_black_box_test(attack_file_path, [vq_hard], y_test, x_test)

    # White box 
    attacks = []
    if args['BIM_white']:
        attacks.append(BasicIterativeMethod)
    if args['FGSM_white']:
        attacks.append(FastGradientMethod)
    if args['CIM_white']:
        attacks.append(CarliniLInfMethod)

    if attacks != []:
        eps = eval(args['eps'])

        if args['cnn_path'] is not None:
            cnn_white_box_test(eps, attacks, [sess_cnn], [classifier_cnn], [cnn.model], ['cnn'], x_test, y_test)

        if args['vq_path'] is not None:
            vq_white_box_test(eps, attacks, [vq_hard.sess], [classifer_e], vq_models, ['VQ'], x_test, y_test
)


def MNIST_generate_black_box_data(**args):
    # Get data for cifar
    x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test('mnist', fashion = args['fashion'])

    # Params
    if args['fashion'] == False:
        params = mnist_param
    else:
        params = fashion_param

    params.cnn['save_path'] = params.save_path + args['cnn_path']
    sess_cnn, cnn = load_cnn(params.cnn, params.cnn['save_path'], load_last=True)
    classifier_cnn = KerasClassifier(model=cnn.model, clip_values=(_min, _max))

    # Generate Black box data
    eps_list = eval(args['eps'])
    classifier = classifier_cnn
    sess = sess_cnn
    attack_name = ['FGSM','BIM']

    if args['validation_file'] == True:
        attack = FastGradientMethod
        x_test_advs = get_adv_samples(attack, eps_list, x_test, classifier, sess)
        if args['fashion'] == True:
            np.save('save_val/fashion_x_val_adv_vgg.npy', x_test_advs[eps_list[0]])
        else:
            np.save('save_val/mnist_x_val_adv_vgg.npy', x_test_advs[eps_list[0]])
    else:
        for index, attack in enumerate([FastGradientMethod, BasicIterativeMethod]):
            x_test_advs = get_adv_samples(attack, eps_list, x_test, classifier, sess)

            if args['fashion'] == False:
                np.save('save_val/attacks/MNIST_' + attack_name[index] + '_copy_x_test_advs.npy',x_test_advs)
            else:
                np.save('save_val/attacks/Fashion_' + attack_name[index] + '_copy_x_test_advs.npy',x_test_advs)


def MNIST(**args):

    # Get data for cifar
    x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test('mnist', fashion = args['fashion'])

    # Params
    if args['fashion'] == False:
        params = mnist_param
    else:
        params = fashion_param

    save_path = params.save_path

    # Load CNN
    if args['cnn_path'] is not None:
        if args['last_name'] is not None:
            sess_cnn, cnn = load_cnn(params.cnn, args['cnn_path'] + args['last_name'], load_last=False)
        else:
            sess_cnn, cnn = load_cnn(params.cnn, args['cnn_path'], load_last=True)
        classifier_cnn = KerasClassifier(model=cnn.model, clip_values=(_min, _max))

    # Load VQ
    if args['vq_path'] is not None:
        path = args['vq_path']
        vq_hard = load_vq(path, config, args['last_name'])
        classifer_e = TFClassifier(input_ph=vq_hard.input_x,  output = vq_hard.pre_e, labels_ph= vq_hard.y_true,clip_values=(_min, _max), sess=vq_hard.sess, loss=vq_hard.loss_c2)
        vq_models = [vq_hard]

    # Black box test
    if args['fashion'] == False:
        name = 'MNIST'
    else:
        name = 'Fashion'

    paths = ['save_val/attacks/'+name+ '_BIM_copy_x_test_advs.npy', 'save_val/attacks/'+name+ '_FGSM_copy_x_test_advs.npy']
    
    for index, black_flag in enumerate([args['BIM_black'], args['FGSM_black']]):
        if black_flag:
            print(paths[index])

            attack_file_path = paths[index]
            if args['cnn_path'] is not None:
                accs = cnn_black_box_test(attack_file_path, cnn, y_test, x_test)
            if args['vq_path'] is not None:
                vq_accs_q, vq_accs_e = vq_black_box_test(attack_file_path, [vq_hard], y_test, x_test)
    
    # White box 
    attacks = []
    if args['FGSM_white']:
        attacks.append(FastGradientMethod)
    if args['BIM_white']:
        attacks.append(BasicIterativeMethod)
    if args['CIM_white']:
        attacks.append(CarliniLInfMethod)

    if attacks != []:
        eps = eval(args['eps'])

        if args['cnn_path'] is not None:
            cnn_white_box_test(eps, attacks, [sess_cnn], [classifier_cnn], [cnn.model], ['cnn'], x_test, y_test)

        if args['vq_path'] is not None:
            vq_white_box_test(eps, attacks, [vq_hard.sess], [classifer_e], vq_models, ['VQ'], x_test, y_test
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help='run which task',default='MNIST_VQ', required=False)
    parser.add_argument("--fashion", type=str2bool, help='fashion mnist?',default='False')
    parser.add_argument("--validation_file",type=str2bool, help='Generate a validation set?', default='False')


    parser.add_argument("--vq_path",type=str, help='vq path', required=False)
    parser.add_argument("--cnn_path",type=str, help='cnn path', required=False)
    parser.add_argument("--last_name",type=str, help='load_last', required=False)

    parser.add_argument("--gpu",type=str, help='GPU', default='0', required=False)

    parser.add_argument("--FGSM_black",type=str2bool, help='FGSM black-box attack', default='False')
    parser.add_argument("--BIM_black",type=str2bool, help='BIM black-box attack', default='False')
    parser.add_argument("--FGSM_black_large",type=str2bool, help='FGSM large black-box attack', default='False')
    parser.add_argument("--BIM_black_large",type=str2bool, help='BIM large black-box attack', default='False')
    
    parser.add_argument("--BIM_white",type=str2bool, help='BIM white attack', default='False')
    parser.add_argument("--FGSM_white",type=str2bool, help='FGSM whiteattack', default='False')
    parser.add_argument("--CIM_white",type=str2bool, help='CIM white attack', default='False')
    parser.add_argument("--eps",type=str, help='eps for white attack', default='False')


    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
