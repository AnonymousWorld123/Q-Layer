import copy

mode = 'vgg16_pool3'
input_shape = (28,28,1)
batch_size = 128
pre_class = 10
length = 50000
save_path = './save_val/fashion_A/'

regularizer = 0.0005
lr = 0.001

heads = 4
num_concept = 128

dense_dim = 64
concept_dim = 64
classifier_dim = 128


cnn = {
    'pre_class': pre_class,
    'classifier_dim':classifier_dim,
    'lr': lr,
    'input_shape':input_shape,
    'mode':mode,
    'batch_size':batch_size,
    'regularizer':regularizer,
    'save_path': save_path + 'cnn/',
    'optimizer':'Adam'
}

cnn_B = copy.copy(cnn)
cnn_B['mode'] = 'B'
cnn_B['classifier_dim'] = 0
cnn_B['save_path'] = save_path + 'cnn_B/'


hard = {
    'pre_class': pre_class,
    'dense_dim': dense_dim, 
    'concept_dim': concept_dim,   
    'classifier_dim':classifier_dim,
    'num_concept': num_concept,
    'c1':1,
    'c2':1,
    'alpha':0.001,
    'beta':0.001,
    'theta': 0,
    'gamma': 0,
    'lr': lr,
    'lr_decay_rate':0.3,
    'lr_decay_steps': 10 * length // batch_size,
    'set_att': heads,
    'set_trans': False,
    'input_shape':input_shape,
    'mode':mode,
    'batch_size': batch_size,
    'distribute_size': None,
    'update': True,
    'update_lr':0.001,
    'save_best':True,
    'regularizer':regularizer,
    'early_stop':10,
    'save_path': save_path + 'hard/',
    'direct_pass': False,
    'omega':0,
    'optimizer':'AdamOptimizer'
}

hard_inter = copy.copy(hard)
# hard_inter['start_layer'] = 'A_start'
# hard_inter['inter_layer'] = 'A_start'
hard_inter['inter_layer'] = 'vgg16_pool0.5'
hard_inter['start_layer'] = 'vgg16_start'
hard_inter['batch_size'] = 128
hard_inter['update'] = 'train'
hard_inter['save_path'] = save_path + 'hard_inter/'
