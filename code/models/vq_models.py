import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from sklearn.manifold import TSNE
from keras.models import Sequential
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.activations import relu, softmax
from keras.models import Model
from keras.metrics import MSE, categorical_crossentropy
import random
from keras import regularizers
from tqdm import tqdm
from six.moves import range
import os
from random import shuffle
from keras.preprocessing import image
from keras.models import load_model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from functools import partial
import time
from models.qlayers import *
from models.basic_model import *
from tensorflow.train import RMSPropOptimizer, AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer
# from keras_radam.training import RAdamOptimizer
import json
from sklearn.cluster import KMeans

cmap = plt.get_cmap('tab20c')

# Load pre-trained VQ model
def load_vq(path, config, last_name = None):
    tf.reset_default_graph()
    f = open(path + '/params.json')
    params_hard = json.load(f)
    params_hard['save_path'] = path
    # params_hard['c1'] =0
    # params_hard['alpha']=0
    # params_hard['beta']=0
    params_hard['lr_decay_steps'] = 30 * 40000 // 128

    vq_hard = VQ_clustering_inter(config = config, params = params_hard) 
    if last_name is not None:
        vq_hard.load(path+last_name)
    else:
        vq_hard.load_lastpoint()
    return vq_hard

def get_logits(vq_hard):
    if 'start_layer' in vq_hard.params:
        logits_q = vq_hard.x_q
        logits_e = vq_hard.x_e
        logits_path = vq_hard.x_e

    else:
        logits_q = vq_hard.z_q
        logits_e = vq_hard.z_e
        logits_path = vq_hard.z_e

    for layer in vq_hard.classifier_q.layers[:-1]:
        logits_q = layer(logits_q)
        logits_path = layer(logits_path)

    for layer in vq_hard.classifier_e.layers[:-1]:
        logits_e = layer(logits_e)

    return(logits_q, logits_e, logits_path)


class VQ_clustering():
    def __init__(self, config, params):
        print(params)
        self.sess = tf.compat.v1.Session(config=config)
        self.pre_class = params['pre_class']
        self.input_shape = params['input_shape']
        self.zeros = tf.constant(0)
        self.params = params
        self.num_spaces = max(params['set_att'],1, params['set_trans'], params['set_fixed'])
        self.num_concept = self.params['num_concept']
        self.build_encoder(mode = params['mode'], input_shape = params['input_shape'])
        self.build_qlayer(set_att = params['set_att'], set_fixed = params['set_fixed'], set_trans = params['set_trans'],dense_dim = params['dense_dim'], concept_dim = params['concept_dim'], num_concept = params['num_concept'], direct_pass = params['direct_pass'])
        self.build_classifier(params['classifier_dim'])
        self.build_forward(input_shape = params['input_shape'])
        self.build_dis()
        self.build_backward(
            c1 = params['c1'], c2 = params['c2'], 
            alpha = params['alpha'],beta = params['beta'], omega = params['omega'],
            theta = params['theta'], gamma = params['gamma'],lr = params['lr'])
        self.build_metric()
        self.build_update_inactive(lr = params['update_lr'])
        self.init()
        self.train_losses = None
        self.train_accs = None
        self.val_loss = None
        self.saver = tf.compat.v1.train.Saver(max_to_keep=100)
        self.init_embeds_flag = False
        self.train_epoch = 0

        print(self.params['save_path'])
        if os.path.exists(self.params['save_path']) == False:
            os.mkdir(self.params['save_path'])
        
        self.save_params()
        
    def save_params(self):
        json_str = json.dumps(self.params)
        with open(self.params['save_path'] + '/params.json', 'w') as json_file:
            json_file.write(json_str)

    def build_encoder(self, mode = '2d', **kwargs):
        self.encoder = get_encoder(mode, **kwargs)

    def build_qlayer(self,**kwargs):
        self.qlayer = QLayer(**kwargs)
    
    def build_classifier(self, classifier_dim = 64):   
        if ('share_backward' not in self.params):
            share = False
        elif self.params['share_backward'] == False:
            share = False
        else:
            share = True
        
        if share == False:
            if 'regularizer' in self.params:
                self.classifier_q = get_classifier(self.pre_class, classifier_dim, self.params['regularizer'])
                self.classifier_e = get_classifier(self.pre_class, classifier_dim, self.params['regularizer'])
            else:
                self.classifier_q = get_classifier(self.pre_class, classifier_dim)
                self.classifier_e = get_classifier(self.pre_class, classifier_dim)
        else:
            if 'regularizer' in self.params:
                self.classifier_q = get_classifier(self.pre_class, classifier_dim, self.params['regularizer'])
            else:
                self.classifier_q = get_classifier(self.pre_class, classifier_dim)
                
            self.classifier_e = self.classifier_q


    def build_forward(self, input_shape = (32,32,3)):
        self.input_x = Input(shape=input_shape)
        self.x = self.encoder(self.input_x)
        if self.params['direct_pass']:
            self.z_q, self.z_e, self.z_ks, self.z_pass = self.qlayer(self.x)
        else:
            self.z_q, self.z_e, self.z_ks = self.qlayer(self.x)
        
        self.pre_q = self.classifier_q(self.z_q)
        self.pre_e = self.classifier_e(self.z_e)
        self.model = Model(self.input_x, [self.pre_q, self.pre_e, self.z_ks])
        self.model_e = Model(self.input_x, self.pre_e)
    

    def build_path(self):
        self.pre_path = self.classifier_q(Flatten()(self.z_e))
        self.loss_path = tf.reduce_mean(K.categorical_crossentropy(self.y_true, self.pre_path))

    def build_dis(self, qlayer = None):
        if qlayer is None:
            qlayer = self.qlayer
        self.active_dis = []
        self.inactive_dis = []
        self.active_node = []

        for i in range(self.num_spaces):
            active = tf.unique(tf.reshape(self.z_ks[i],shape=(-1,)), out_idx=tf.int64)[0]
            self.active_node.append(active)
            active_embeds = tf.gather(qlayer.embeds[i], active)
            active_center = tf.reduce_mean(active_embeds, axis = 0)
            active_v = tf.reduce_mean(tf.reduce_sum((active_embeds - active_center)**2, axis = -1))
            self.active_dis.append(active_v)
    

    def update_inactive(self, actives, es, random_flag = False, kmeans_flag = False, qlayer = None):
        if qlayer is None:
            qlayer = self.qlayer
        embs = self.sess.run(qlayer.embeds)

        if ('update_strategy' in self.params) and (self.params['update_strategy'] == 'update_to_concept'):
            _to_concept = True
            print('update to closest concept')
        else:
            _to_concept = False
        
        for i in range(self.num_spaces):
            emb = embs[i]
            active = actives[i]
            
            if _to_concept:
                e = emb[active]
            else:
                e = es[i]
            inactive = np.logical_not(active)
        
            if (random_flag == False) and (kmeans_flag == False):
                # Not used
                inactive_embeds = emb[inactive]
                _t = np.expand_dims(inactive_embeds, axis=-2)
                _t = np.sum((_t - e)**2, axis = -1)
                _k = np.argmin(_t, axis = -1)
                emb[inactive] = e[_k] #+ np.random.normal(loc=0, scale = 0.01, size = e[_k].shape)
            
            elif random_flag == True:
                # Randomly init
                _k = random.sample(list(range(len(e))), len(inactive))
                emb[inactive] = e[_k]
            
            elif kmeans_flag == True:
                # Kmeans center init
                print('Running Kmeans init')
                k = self.num_concept
                centroids = KMeans(k, n_init = 1).fit(e).cluster_centers_
                emb[inactive] = centroids
            
            update_op = tf.assign(qlayer.embeds[i], emb)
            _ = self.sess.run(update_op)

    def build_update_inactive(self, lr):
        # Force a concept to move to its cloesest data point
        self.input_active = tf.compat.v1.placeholder(tf.bool,[self.num_spaces,self.num_concept])
        self.inactive_dis = []

        for i in range(self.num_spaces):
            active = self.input_active[i]
            active_embeds = tf.boolean_mask(self.qlayer.embeds[i], active,axis=0)
            active_center = tf.reduce_mean(active_embeds, axis = 0)
            inactive = tf.logical_not(active)
            inactive_embeds = tf.boolean_mask(self.qlayer.embeds[i], inactive, axis=0)

            _t = tf.expand_dims(inactive_embeds, axis=-2)
            _t = tf.reduce_sum((_t - active_embeds)**2, axis = -1)
            inactive_dis = tf.reduce_mean(tf.reduce_min(_t,axis=-1))
            self.inactive_dis.append(inactive_dis)
        
        self.update_lr = tf.train.exponential_decay(lr, self.global_step, decay_steps = self.params['lr_decay_steps'],decay_rate = self.params['lr_decay_rate'])
        self.loss_inactive = tf.reduce_mean(self.inactive_dis)
        update_optimizer = tf.train.AdamOptimizer(self.update_lr)
        self.update_op= update_optimizer.minimize(self.loss_inactive)
    
    def build_backward(self, c1 =1, c2 = 1, alpha = 0.01, beta = 0.01, omega = 0.01, theta = 10, gamma = 0.01, lr = 0.0001):
        self.y_true = tf.compat.v1.placeholder(tf.float32,[None, self.pre_class])

        self.loss_c1 = tf.reduce_mean(K.categorical_crossentropy(self.y_true, self.pre_q))
        self.loss_c2 = tf.reduce_mean(K.categorical_crossentropy(self.y_true, self.pre_e))
        self.loss_c = c1 * self.loss_c1 + c2 * self.loss_c2

        self.loss_btclass = -tf.reduce_mean(self.active_dis)
        # self.loss_vq = tf.reduce_mean(
        #     tf.reduce_sum((tf.stop_gradient(self.z_e)- self.z_q)**2,axis = -1))
        # self.loss_commit = tf.reduce_mean(
        #     tf.reduce_sum(self.z_e - tf.stop_gradient(self.z_q)**2, axis = -1)) 

        self.loss_vq = tf.reduce_mean(
            tf.reduce_sum((self.z_e- self.z_q)**2,axis = -1))

        if ((c1 == 0) and (alpha == 0) and (beta == 0)):
            self.loss = self.loss_c2
            print('Only optimize E-path')
            flag = False

        else:
            flag = True
            
            if self.params['direct_pass'] == True:
                #self.loss_pass_threshold = -tf.norm(self.qlayer.pass_threshold)
                self.loss = self.loss_c + alpha * self.loss_vq #+ beta * self.loss_commit #+ theta * self.loss_btclass #+ gamma * self.loss_inacitve
            else:
                self.loss = self.loss_c + alpha * self.loss_vq #+ beta * self.loss_commit
                self.loss_e = self.loss_c2
            
            if 'alpha_trans' in self.params:
                alpha_trans = self.params['alpha_trans']
                beta_trans = self.params['beta_trans']
            else:
                alpha_trans = alpha
                beta_trans = beta
            
            self.trans_loss = alpha_trans * self.loss_vq #+ beta_trans * self.loss_commit

            
        self.global_step = tf.Variable(0, name='global_step',trainable=False)

        if 'lr_decay_steps' in self.params:
            print('Use adaptive learning rate')
            self.lr = tf.compat.v1.train.exponential_decay(self.params['lr'], self.global_step, decay_steps = self.params['lr_decay_steps'],decay_rate = self.params['lr_decay_rate'])
            print('Use optimizer',self.params['optimizer'])
            optimizer = eval(self.params['optimizer'])(self.lr)

        else:
            optimizer = tf.train.AdamOptimizer(self.params['lr'])

        self.train_op = optimizer.minimize(self.loss,global_step=self.global_step)

        if ((c1 == 0) and (alpha == 0) and (beta == 0)):
            self.trans_op = self.train_op
        else:
            self.trans_op = optimizer.minimize(self.trans_loss,global_step=self.global_step)
    
    def reset_global_step(self):
        ass = self.global_step.assign(0)
        self.sess.run(ass)
        print('Reset global step',self.sess.run(self.global_step))

    def build_metric(self):
        correct_prediction_q = tf.equal(tf.argmax(self.pre_q, 1), tf.argmax(self.y_true, 1))
        self.accuracy_q = tf.reduce_mean(tf.cast(correct_prediction_q, tf.float32))
        correct_prediction_e = tf.equal(tf.argmax(self.pre_e, 1), tf.argmax(self.y_true, 1))
        self.accuracy_e = tf.reduce_mean(tf.cast(correct_prediction_e, tf.float32))

    def init(self):
        print("Initing Session")
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

    def init_embeds(self, X, random_flag = True, kmeans_flag = False):
        '''
        Only used to init embeds before training
        '''
        actives = np.zeros((self.num_spaces, self.num_concept),bool)
        es = self.sess.run(self.z_e, feed_dict={self.input_x:X})
        es = np.array(es).reshape(-1, self.num_spaces, self.params['concept_dim']).transpose(1,0,2)

        if (random_flag):
            self.update_inactive(actives, es, random_flag = random_flag)
        elif kmeans_flag:
            self.update_inactive(actives, es, random_flag = False, kmeans_flag = True)

        self.init_embeds_flag = False
        print('Init embeds')

    def run_epoch(self, train_X=None,train_Y=None,val_X=None, val_Y=None, train_flow=None, step_per_epoch = None, mode='train', BATCH_SIZE = 32, summary_it = 1000, attacks = None, attack_ratio = None, trans_X = None, trans_Y = None):
        sess = self.sess
        train_losses1 = []
        train_losses2 = []
        train_losses3 = []
        train_losses4 = []
        train_losses = []
        train_accs_q = []
        train_accs_e = []
        actives = np.zeros((self.num_spaces, self.num_concept),bool)
        attack_id = 0
        print_loss = self.loss_c2

        if 'train' in mode:
            print('lr',self.sess.run(self.lr))


        if trans_X is not None:
            trans_losses = []
            trans_accs_q = []
            trans_accs_e = []
            trans_actives = np.zeros((self.num_spaces, self.num_concept),bool)

        if ('train' in mode):
            X = train_X
            Y = train_Y
            t1 = time.time()
        else:
            X = val_X
            Y = val_Y
        
        if trans_X is not None:
            print('Get trans batch generator')
            def trans_batch(trans_X, trans_Y):
                step = 0
                indexs = list(range(0,trans_X.shape[0]))
                np.random.shuffle(indexs)

                while True:
                    batch_xs = trans_X[indexs][step:step+BATCH_SIZE]
                    batch_ys = trans_Y[indexs][step:step+BATCH_SIZE]
                    step = step + BATCH_SIZE

                    if step > trans_X.shape[0]:
                        step = 0
                        np.random.shuffle(indexs)

                    yield batch_xs, batch_ys

            gen = trans_batch(trans_X, trans_Y)


        for step in range(0, len(X), BATCH_SIZE):  
            if ((train_flow is not None) and ('train' in mode)):
                batch_xs, batch_ys = train_flow[step // BATCH_SIZE]
            else:
                batch_xs = X[step:step+BATCH_SIZE]
                batch_ys = Y[step:step+BATCH_SIZE]

            if trans_X is not None:
                trans_batch_xs, trans_batch_ys = next(gen)
            
            # Adv training
            if mode == 'adv_train':
                attack = attacks[attack_id]
                nb_adv = int(np.ceil(attack_ratio * batch_xs.shape[0]))

                if attack_ratio < 1:
                    adv_ids = np.random.choice(batch_xs.shape[0], size=nb_adv, replace=False)
                else:
                    adv_ids = list(range(batch_xs.shape[0]))
                    # np.random.shuffle(adv_ids)

                batch_xs[adv_ids] = attack.generate(batch_xs[adv_ids])

                attack_id = attack_id + 1
                if attack_id == len(attacks):
                    attack_id = 0
 

            if (('train' in mode) and (self.init_embeds_flag == 'random')):
                # Before the first epoch, init embeds
                self.init_embeds(batch_xs, random_flag = True)

            elif (('train' in mode) and (self.init_embeds_flag == 'kmeans')):
                # Before the first epoch, init embeds
                self.init_embeds(batch_xs, random_flag = False, kmeans_flag = True)

            if ('train' in mode):
                it, train_loss1, train_loss2, train_loss3, train_loss4, train_loss, train_acc_q, train_acc_e, active, _ = sess.run(
                    [self.global_step, self.loss_c, self.loss_vq, self.loss_btclass, print_loss,self.loss, self.accuracy_q,self.accuracy_e, self.active_node, self.train_op], 
                    feed_dict={self.input_x:batch_xs, self.y_true:batch_ys})

                if trans_X is not None:
                    it, trans_loss, trans_acc_q, trans_acc_e, trans_active, _ = sess.run(
                        [self.global_step, self.loss_vq, self.accuracy_q,self.accuracy_e, self.active_node, self.trans_op], 
                        feed_dict={self.input_x:trans_batch_xs, self.y_true:trans_batch_ys})
            else:
                train_loss1, train_loss2, train_loss3, train_loss4, train_loss, train_acc_q, train_acc_e, active = sess.run(
                    [self.loss_c, self.loss_vq, self.loss_btclass, print_loss, self.loss, self.accuracy_q,self.accuracy_e, self.active_node], 
                    feed_dict={self.input_x:batch_xs, self.y_true:batch_ys})
            
            for j in range(self.num_spaces):
                ac = np.unique(active[j])
                actives[j][ac] = True
            
            if trans_X is not None:
                trans_losses.append(trans_loss.mean())
                trans_accs_q.append(trans_acc_q.mean())
                trans_accs_e.append(trans_acc_e.mean())

                for j in range(self.num_spaces):
                    ac = np.unique(trans_active[j])
                    trans_actives[j][ac] = True

            
            if (('train' in mode) and (self.params['update'] == 'batch')):
                es = sess.run(self.z_e, feed_dict={self.input_x:batch_xs})
                es = np.array(es).reshape(-1, self.num_spaces, self.params['concept_dim']).transpose(1,0,2)
                self.update_inactive(actives, es)
                print('Update inactive in batch')

            train_accs_q.append(train_acc_q)
            train_accs_e.append(train_acc_e)
            train_losses1.append(train_loss1.mean())
            train_losses2.append(train_loss2.mean())
            train_losses3.append(train_loss3.mean())
            train_losses4.append(train_loss4.mean())
            train_losses.append(train_loss.mean())

            if ('train' in mode):
                if summary_it is not None:
                    if it % summary_it == 0:
                        t2 = time.time()
                        t = t2-t1
                        t1 = time.time()

                        print('[%5d] Time [%4.1f] Loss_c: %1.4f Loss_vq: %1.4f Loss_bt: %1.4f Loss 4: %1.4f Loss: %1.4f Acc_q: %2.4f Acc_e: %2.4f'%(
                        it,t,np.mean(train_losses1), np.mean(train_losses2), np.mean(train_losses3), np.mean(train_losses4), np.mean(train_losses),np.mean(train_accs_q), np.mean(train_accs_e)))
                        if trans_X is not None:
                            print('Trans Loss_vq: %1.4f Acc_q: %2.4f Acc_e: %2.4f'%(np.mean(trans_losses), np.mean(trans_accs_q), np.mean(trans_accs_e)))
        
        avg = len(np.where(actives)[0]) / self.num_spaces
        print(mode,'average active nodes:',avg)

        if trans_X is not None:
            trans_avg = len(np.where(trans_actives)[0]) / self.num_spaces
            print('trans average active nodes:',trans_avg)

        if ('train' in mode):
            # Update inactive
            if (self.params['update'] == 'train'):
                es = sess.run(self.z_e, feed_dict={self.input_x:batch_xs})
                es = np.array(es).reshape(-1, self.num_spaces, self.params['concept_dim']).transpose(1,0,2)
                self.update_inactive(actives, es)
                print('Update inactive in train')

            elif (self.params['update'] == True):
                _, loss_inactive = sess.run([self.update_op, self.loss_inactive],feed_dict={self.input_active:actives})
                print('Optimize inactive distance, Loss_inactive:',loss_inactive)
      
            # Val
            val_loss1, val_loss2, val_loss3, val_loss4,val_loss, val_acc_q, val_acc_e, val_avg = self.run_epoch(train_X,train_Y,val_X,val_Y, mode='val',train_flow=None)
            self.val_loss.append(val_loss.mean())
            self.val_loss1.append(val_loss1.mean())
            self.val_loss2.append(val_loss2.mean())
            self.val_loss3.append(val_loss3.mean())
            self.val_loss4.append(val_loss4.mean())
            self.val_acc_q.append(val_acc_q.mean())
            self.val_acc_e.append(val_acc_e.mean())
            

            # Print
            print('[%5d] Train Loss: %1.4f Loss_c: %1.4f Loss_vq: %1.4f Loss_bt: %1.4f  Acc_q: %2.4f Acc_e: %2.4f'%(
                it,np.mean(train_losses), np.mean(train_losses1), np.mean(train_losses2), np.mean(train_losses3),np.mean(train_accs_q), np.mean(train_accs_e)))
        
            print(
                '[%5d] Val Loss: %1.4f Loss_c: %1.4f Loss_vq: %1.4f Loss_bt: %1.4f Acc_q: %2.4f Acc_e: %2.4f'%(
                it, np.mean(val_loss),np.mean(val_loss1),np.mean(val_loss2), np.mean(val_loss3), np.mean(val_acc_q), np.mean(val_acc_e)))

            # Save best model
            if ('save_best' in self.params):

                if (type(self.params['save_best']) != int):
                    if self.params['save_best'] == True:
                        compare = val_acc_q.mean() == np.max(self.val_acc_q)
                    elif self.params['save_best'] == 'sum':
                        compare = ((val_acc_q+ val_acc_e).mean()) == (np.max(self.val_acc_q+self.val_acc_e))
                    elif self.params['save_best'] == 'q':
                        compare = val_acc_q.mean() == np.max(self.val_acc_q)
                    elif self.params['save_best'] == 'e':
                        compare = val_acc_e.mean() == np.max(self.val_acc_e)

                    if compare:
                        try:
                            s = self.params['save_path']+str(self.train_epoch)+'_'+str(it)+'_q_'+str(val_acc_q.mean())[:6]+'_e_'+str(val_acc_e.mean())[:6]+'_loss_'+str(val_loss.mean())[:6]
                        except:
                            s = './save/hard'+str(self.train_epoch)+'_'+str(it)+'_q_'+str(val_acc_q.mean())[:6]+'_e_'+str(val_acc_e.mean())[:6]+'_loss_'+str(val_loss.mean())[:6]
                        print('Save best model at',s)
                        self.save(s)

                elif type(self.params['save_best']) == int:
                    if self.train_epoch % self.params['save_best'] == 0:
                        s = self.params['save_path']+str(self.train_epoch)+'_'+str(it)+'_q_'+str(val_acc_q.mean())[:6]+'_e_'+str(val_acc_e.mean())[:6]+'_loss_'+str(val_loss.mean())[:6]
                        print('Save model at',s)
                        self.save(s)      
            
            # Early stop
            if ('early_stop' in self.params) and (self.params['early_stop'] > 0):
                if len(self.val_loss) > self.params['early_stop']:
                    max_pos = np.argsort(self.val_acc_q)[-1]
                    if self.params['early_stop'] < len(self.val_loss)-1-max_pos:
                        self.early_stop = True
            
            return(it, np.array(train_losses1), np.array(train_losses2), np.array(train_losses3), np.array(train_losses4),np.array(train_losses), np.array(train_accs_q),np.array(train_accs_e), avg)

        elif mode == 'val':
            # Update inactive
            if (self.params['update'] == 'val'):

                es = sess.run(self.z_e, feed_dict={self.input_x:batch_xs})
                es = np.array(es).reshape(-1, self.num_spaces, self.params['concept_dim']).transpose(1,0,2)
                self.update_inactive(actives, es)
                print('Update inactive in val')

            return(np.array(train_losses1), np.array(train_losses2), np.array(train_losses3), np.array(train_losses4),np.array(train_losses), np.array(train_accs_q),np.array(train_accs_e),avg)
        
        elif mode == 'test':
            return(np.array(train_losses1), np.array(train_losses2), np.array(train_losses3), np.array(train_losses4),np.array(train_losses), np.array(train_accs_q),np.array(train_accs_e),avg)

    def train(self, train_X,train_Y,val_X, val_Y, train_flow = None, step_per_epoch = None, TRAIN_EPOCH = 15, BATCH_SIZE = 32, shuffle = True, summary_it = None, attacks = None, attack_ratio = None, trans_X = None, trans_Y = None):
        print('Start training')
        train_losses1 = []
        train_losses2 = []
        train_losses3 = []
        train_losses4 = []
        train_losses = []
        train_accs_q = []
        train_accs_e = []
        train_avgs = []

        if self.val_loss is None:
            self.val_loss = []
            self.val_loss1 = []
            self.val_loss2 = []
            self.val_loss3 = []
            self.val_loss4 = []
            self.val_acc_q = []
            self.val_acc_e = []
        
        self.early_stop = False
        for epoch in range(TRAIN_EPOCH):
            print('-'*30,'EPOCH',epoch,'-'*30)
            self.train_epoch = self.train_epoch + 1

            if train_flow is not None:
                if ((attacks is not None) and (attack_ratio is not None)):
                    it, train_loss1, train_loss2, train_loss3, train_loss4, train_loss, train_acc_q, train_acc_e, avg = self.run_epoch(train_X, train_Y,val_X, val_Y, train_flow = train_flow, step_per_epoch = step_per_epoch, mode = 'adv_train', BATCH_SIZE = BATCH_SIZE, summary_it = summary_it, attacks =attacks, attack_ratio = attack_ratio, trans_X = trans_X, trans_Y = trans_Y)
                
                else:
                    it, train_loss1, train_loss2, train_loss3, train_loss4, train_loss, train_acc_q, train_acc_e, avg = self.run_epoch(train_X, train_Y,val_X, val_Y, train_flow = train_flow, step_per_epoch = step_per_epoch, mode = 'train', BATCH_SIZE = BATCH_SIZE, summary_it = summary_it, trans_X = trans_X, trans_Y = trans_Y)
            
            else:                
                indexs = list(range(len(train_X)))
                if (shuffle):
                    np.random.shuffle(indexs)

                indexs2 = list(range(len(val_X)))
                if (shuffle):
                    np.random.shuffle(indexs2)
                
                if (attacks is not None) and (attack_ratio is not None):
                    it, train_loss1, train_loss2, train_loss3, train_loss4, train_loss, train_acc_q, train_acc_e, avg = self.run_epoch(train_X = train_X[indexs], train_Y = train_Y[indexs],val_X = val_X[indexs2], val_Y = val_Y[indexs2], mode = 'adv_train', BATCH_SIZE = BATCH_SIZE, summary_it = summary_it, attacks =attacks, attack_ratio = attack_ratio, trans_X = trans_X, trans_Y = trans_Y)
                
                else:
                    it, train_loss1, train_loss2, train_loss3, train_loss4, train_loss, train_acc_q, train_acc_e, avg = self.run_epoch(train_X = train_X[indexs], train_Y = train_Y[indexs],val_X = val_X[indexs2], val_Y = val_Y[indexs2], mode = 'train', BATCH_SIZE = BATCH_SIZE, summary_it = summary_it, trans_X = trans_X, trans_Y = trans_Y )
            
            train_losses1.append(train_loss1.mean())
            train_losses2.append(train_loss2.mean())
            train_losses3.append(train_loss3.mean())
            train_losses4.append(train_loss4.mean())
            train_losses.append(train_loss.mean())
            train_accs_q.append(train_acc_q.mean())
            train_accs_e.append(train_acc_e.mean())
            train_avgs.append(avg)
            
            if self.early_stop == True:
                print('Early stopping at epoch', epoch)
                break
        
        print('End training')

        if (self.train_losses is None):
            self.train_losses = [train_losses1, train_losses2, train_losses3, train_losses4, train_losses]
            self.train_accs = [train_accs_q, train_accs_e]
            self.train_avgs = train_avgs

        else:
            self.train_losses = [self.train_losses[0] + train_losses1, self.train_losses[1] + train_losses2, self.train_losses[2] + train_losses3, self.train_losses[3] + train_losses4, self.train_losses[4] + train_losses]
            self.train_accs = [self.train_accs[0]  + train_accs_q, self.train_accs[1]  + train_accs_e]
            self.train_avgs = self.train_avgs + train_avgs
        
        return(train_losses1, train_losses2, train_losses3, train_losses4, train_losses, train_accs_q, train_accs_e, train_avgs)

    def train_E_path(self, train_X,train_Y,val_X, val_Y, train_flow = None, step_per_epoch = None, TRAIN_EPOCH = 10, BATCH_SIZE = 128, shuffle = True, callbacks = [], attacks = None, attack_ratio = None):
        self.model_e.compile(optimizer='Adam',
                        loss='categorical_crossentropy', 
                        metrics=['accuracy'])
        
        if train_flow is not None:
            hist = self.model_e.fit_generator(train_flow, validation_data=[val_X,val_Y], epochs = TRAIN_EPOCH,
                    steps_per_epoch = train_X.shape[0]//BATCH_SIZE, shuffle=shuffle, callbacks = callbacks)
        
        else:
            hist = self.model_e.fit(train_X, train_Y, batch_size=BATCH_SIZE, validation_data=[val_X,val_Y], epochs = TRAIN_EPOCH,
                  shuffle=shuffle, callbacks = callbacks)
        # model_e.save_weights(save_path + '/4_128_space_10_epoch_pool3_e.ckpt')

    def evaluate(self,val_X, val_Y,BATCH_SIZE = 32, summary_it = None, mode = 'test'):
        train_loss1, train_loss2, train_loss3, train_loss4, train_loss, train_acc_q, train_acc_e, avg = self.run_epoch(None, None,val_X, val_Y, mode=mode, BATCH_SIZE=BATCH_SIZE, summary_it=summary_it)
        return(train_loss1.mean(), train_loss2.mean(), train_loss3.mean(), train_loss4.mean(),train_loss.mean(), train_acc_q.mean(), train_acc_e.mean(), avg)


    def compute_actives(self, X, Y, BATCH_SIZE = 32):
        actives = np.zeros((self.num_spaces, self.num_concept),bool)
        
        for step in range(0, len(X), BATCH_SIZE):
            batch_xs = X[step:step+BATCH_SIZE]
            batch_ys = Y[step:step+BATCH_SIZE]
            active = self.sess.run(self.active_node, feed_dict={self.input_x:batch_xs})
            
            for j in range(self.num_spaces):
                ac = np.unique(active[j])
                actives[j][ac] = True
        
        self.actives = actives
        
        return(actives)


    # def evaluate_with_actives(self, X, Y, BATCH_SIZE = 32, actives = None):
    #     if actives is None:
    #         actives = self.actives
        
    #     accs_q = []
    #     accs_e = []
        
    #     for step in range(0, len(X), BATCH_SIZE):
    #         batch_xs = X[step:step+BATCH_SIZE]
    #         batch_ys = Y[step:step+BATCH_SIZE]
    #         acc_q, acc_e = self.sess.run([self.active_accuracy_q, self.active_accuracy_e], feed_dict={self.input_x:batch_xs, self.input_actives:actives, self.y_true:batch_ys})
    #         accs_q.append(acc_q)
    #         accs_e.append(acc_e)
        
    #     return(np.mean(accs_q),np.mean(accs_e))
    
    def save(self, path):
        self.saver.save(self.sess,path)
    
    def load(self,path):
        self.saver.restore(self.sess, path)
    
    def load_lastpoint(self, path = None):
        if path == None:
            path = tf.train.latest_checkpoint(self.params['save_path'])
        else:
            print('Load last checkpoint from',path)
            path = tf.train.latest_checkpoint(path)
       
        print('Load last checkpoint from',path)
        
        self.load(path)
        self.init_embeds_flag = False
    

    def find_best_lastpoint(self, x_val, y_val, path = None, batch_size = 128):
        if path is None:
            path = self.params['save_path']
        
        paths = tf.train.get_checkpoint_state(path).all_model_checkpoint_paths
        best_acc_q = -1
        best_path = paths[0]
        for ck_path in paths:
            self.load(ck_path)
            acc = self.evaluate(x_val,y_val,BATCH_SIZE=batch_size)[-3:-1]
            acc_q = round(acc[0]*100,2)
            acc_e = round(acc[1]*100,2)
            if acc_q > best_acc_q:
                best_path = ck_path
                best_acc_q = acc_q
                print(acc_q, acc_e)
        
        f = open(path + 'best_model','w')
        d = f.write(best_path)
        f.close()
        print(best_path)
        self.init_embeds_flag = False

    def load_best_lastpoint(self, path = None):
        if path is None:
            path = self.params['save_path']
       
        file_path = path + 'best_model'
        
        if os.path.exists(file_path):
            best_path = open(path + 'best_model','r').readlines()[0]
            self.load(best_path)
            print(best_path)
        else:
            self.load_lastpoint(path)
        
        self.init_embeds_flag = False

    def run_variable(self, X, variable, BATCH_SIZE = 32):
        sess = self.sess
        train = []
        
        for step in range(0, len(X), BATCH_SIZE):
            batch_xs = X[step:step+BATCH_SIZE]
            z_ks = sess.run(variable, feed_dict={self.input_x:batch_xs})
            train.append(z_ks)
        return(train)


class VQ_clustering_inter(VQ_clustering):
    def __init__(self, config, params):
        self.inter_layer = params['inter_layer']
        VQ_clustering.__init__(self, config, params) 
    
    def build_encoder(self, mode = '2d',**kwarg):
        self.encoder = get_inter_encoder(start=self.params['start_layer'], end = self.params['inter_layer'], input_shape = self.params['input_shape'])
        
    def build_forward(self, input_shape = (32,32,3)):
        self.input_x = Input(shape = input_shape)
        x = self.input_x

        # encoder 1
        self.x1 = self.encoder(x)

        # Adding qlayer
        if self.params['direct_pass']:
            self.z_q, self.z_e, self.z_ks, self.z_pass = self.qlayer(self.x1)
        else:
            self.z_q, self.z_e, self.z_ks = self.qlayer(self.x1)
        
        # encoder 2
        shape = self.qlayer.output_shape[0][1:]
        print('Build inter layer at', self.params['inter_layer'])

        if ('share_backward' in self.params):
            if self.params['share_backward'] == True:
                share = True
            else:
                share = False
        else:
            share = False

        self.encoder_q = get_inter_encoder(start=self.params['inter_layer'], end = self.params['mode'], input_shape=shape)

        if share:
            self.encoder_e = self.encoder_q
        else:
            self.encoder_e = get_inter_encoder(start=self.params['inter_layer'], end = self.params['mode'], input_shape=shape)
        
        self.x_e = self.encoder_e(self.z_e)
        self.x_q = self.encoder_q(self.z_q)
        self.pre_q = self.classifier_q(self.x_q)
        self.pre_e = self.classifier_e(self.x_e)
        self.model = Model(self.input_x, [self.pre_q, self.pre_e])
    
    def build_path(self):
        self.pre_path = self.classifier_q(self.encoder_q(self.z_e))
        self.loss_path = tf.reduce_mean(K.categorical_crossentropy(self.y_true, self.pre_path))


class vq_clustering_multi_inter(VQ_clustering):
    '''
    Not completed. 
    To do: kmeans init and inactive update.
    '''
    def __init__(self, config, params):
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)
        self.params = params

    def build_qlayer(self):
        params = self.params
        self.num_qlayer = len(params['inter_layer'])
        self.qlayers = []

        for i in range(self.num_qlayer):
            qlayer = QLayer(set_att = params['set_att'][i], set_fixed = params['set_fixed'][i], set_trans = params['set_trans'][i],dense_dim = params['dense_dim'][i], concept_dim = params['concept_dim'][i], num_concept = params['num_concept'][i], direct_pass = params['direct_pass'])
            self.qlayers.append(qlayer)
    
      
    def build_encoder(self, mode = '2d',**kwarg):
        input_shape = self.params['input_shape']
        for i in range(len(self.params['inter_layer'])):
            encoder = get_inter_encoder(start=self.params['start_layer'][i], end = self.params['inter_layer'][i], input_shape = input_shape)
        
    def build(self, loss_weight = 1e-3):
        input_shape = self.params['input_shape']
        self.input_x = Input(shape = input_shape)
        x_q = self.input_x
        x_e = self.input_x

        self.num_qlayer = len(params['inter_layer'])
        self.qlayers = []
        self.encoders = []
        self.loss_qs = []

        for i in range(self.num_qlayer):
            print('Build inter layer at', self.params['inter_layer'])

            encoder = get_inter_encoder(start=self.params['start_layer'][i], end = self.params['inter_layer'][i], input_shape = input_shape)
            self.encoders.append(encoder)

            qlayer = QLayer(set_att = params['set_att'][i], set_fixed = params['set_fixed'][i], set_trans = params['set_trans'][i],dense_dim = params['dense_dim'][i], concept_dim = params['concept_dim'][i], num_concept = params['num_concept'][i], direct_pass = params['direct_pass'])
            self.qlayers.append(qlayer)
            
            # Forward 
            x_q = self.encoder(x_q)
            x_e = self.encoder(x_e)
            x_q, x_e, z_ks = self.qlayer(self.x_q)

            # Loss
            loss_q = tf.reduce_mean(
            tf.reduce_sum((x_q- x_e)**2,axis = -1))
            self.loss_qs.append(loss_q)

            input_shape = self.qlayer.output_shape[0][1:]

        
        self.pre_q = self.classifier_q(x_q)
        self.pre_e = self.classifier_e(x_e)


        self.y_true = tf.compat.v1.placeholder(tf.float32,[None, self.pre_class])
        self.loss_c1 = tf.reduce_mean(K.categorical_crossentropy(self.y_true, self.pre_q))
        self.loss_c2 = tf.reduce_mean(K.categorical_crossentropy(self.y_true, self.pre_e))
        self.loss_c = c1 * self.loss_c1 + c2 * self.loss_c2

        self.loss_vq = tf.reduce_mean(self.loss_qs)
        self.loss_commit = self.loss_vq
        self.loss = self.loss_c + self.params['alpha'] * self.loss_vq
        self.loss_e = self.loss_c2

        # self.model = Model(self.input_x, [self.pre_q, self.pre_e])
        # self.model.add_loss(loss_weight * tf.reduce_mean(self.loss_qs))
        # self.model.compile('Adam', loss=['categorical_crossentropy','categorical_crossentropy'], metrics = ['acc','acc'], metrics_names = ['q_acc', 'e_acc'])
    