import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.pyplot as plt
import os
import random
import keras.backend as K




def get_vq_accs(models, x_test_advs, y_test):
    vq_accs_q = []
    vq_accs_e = []
    for model in models[:]:
        accs_q = []
        accs_e = []
        for scale in x_test_advs:
            x_test_adv = x_test_advs[scale]
            acc = model.evaluate(x_test_adv, y_test)[-3:-1]
            accs_q.append(round(acc[0]*100,2))
            accs_e.append(round(acc[1]*100,2))
            print(scale, str(accs_q[-1]) + '/' + str(accs_e[-1]))
        vq_accs_q.append(accs_q)
        vq_accs_e.append(accs_e)
    return(vq_accs_q, vq_accs_e)

def get_cnn_accs(cnn, x_test_advs, y_test):
    scales = []
    accs = []
    for scale in x_test_advs:
        x_test_adv = x_test_advs[scale]
        scales.append(scale)
        with cnn.sess.as_default():
            with cnn.sess.graph.as_default():
                acc = cnn.model.evaluate(x_test_adv, y_test, verbose = 0)[-1]
                accs.append(round(acc*100,2))
                print(scale, accs[-1])
    return(accs)


def adv_defense(x_test_adv, y_test, sessions, classifiers, names, logger):
    for index,classifier in enumerate(classifiers):
        with sessions[index].as_default():
            with sessions[index].graph.as_default():
                print('-'*30)
                print(names[index])
                preds = np.argmax(classifier.predict(x_test_adv, batch_size=32), axis=1)
                acc = np.sum(preds == np.argmax(y_test, axis=1)) / len(x_test_adv)
                print('Accuracy on adversarial samples:',acc * 100)
    
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
